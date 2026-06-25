import json
import time
from typing import Any, Dict, Optional

import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from mint_plus.utils.log import get_logger
from mint_plus.models.esm2 import ESM2
from mint_plus.models import MODEL_REGISTRY

logger = get_logger(__name__)


class ESMWrapper(pl.LightningModule):
    def __init__(
        self,
        model: ESM2,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.per_chain_rope = self.model_config.get("per_chain_rope", False)
        self.layer_spec = self.model_config.get("layer_spec", None)

        config = MODEL_REGISTRY[self.model_config.get("size", "8M")]
        num_layers=config["num_layers"],

        if not self.layer_spec:
            print(num_layers)
            self.layer_spec = ['self'] * num_layers[0]

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        loss = self._shared_step(batch)

        # Show global_step in progress bar
        self.log("global_step", self.global_step, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        self.stage = "val"
        loss = self._shared_step(batch)
        return loss

    def _shared_step(self, batch):
        """Common forward logic for both training and validation."""
        tokens, chain_ids = batch
        cls_idx = 0          # CLS token index
        eos_idx = self.model.eos_idx
        padding_idx = self.model.padding_idx
        mask_idx = self.model.mask_idx

        # Create mask of non-special positions
        mask = (
            (~tokens.eq(cls_idx))
            & (~tokens.eq(eos_idx))
            & (~tokens.eq(padding_idx))
        )

        # Randomly select 15% of non-special positions to mask
        mask = (torch.rand(tokens.shape, device=tokens.device) < 0.15) & mask

        # Generate random amino acid replacements
        rand = torch.rand(tokens.shape, device=tokens.device)
        randaa = torch.randint(4, 24, tokens.shape, device=tokens.device)

        # Apply masking protocol
        inp = tokens.clone()
        inp = torch.where((rand < 0.8) & mask, mask_idx, inp)
        inp = torch.where((rand > 0.9) & mask, randaa, inp)
        
        # Forward pass
        out = self.model(inp, chain_ids=chain_ids)["logits"]

        # Compute cross-entropy loss averaged over masked positions
        loss = F.cross_entropy(out.transpose(1, 2), tokens, reduction="none")
        loss = (loss * mask).sum() / mask.sum()

        # Log metrics - automatically aggregated across DDP ranks
        self.log(f"{self.stage}_loss", loss, sync_dist=True, prog_bar=True)
        self.log(f"{self.stage}_perplexity", torch.exp(loss), sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        lr = self.training_config.get("lr", 1e-4)
        betas = json.loads(self.training_config.get("adam_betas", "[0.9, 0.98]"))
        eps = self.training_config.get("adam_eps", 1e-8)
        weight_decay = self.training_config.get("weight_decay", 0.01)
        freeze_attn = self.training_config.get("freeze_self_attn", False)
        
        if self.training_config.get('fp8', False):
            logger.info('Using fp8 optimizer configuration')
            optimizer = self._configure_fp8_optimizers(lr, betas, eps, weight_decay, freeze_attn)

        else:
            # non fp8 training (non munit scaling)
            if freeze_attn:
                self.model.requires_grad_(False)
    
                cross_attn_indices = [str(i) for i, x in enumerate(self.layer_spec) if x == 'cross']
                for name, p in self.model.named_parameters():
                    layer_idx = name.split('.')[2]  # after _orig_mod. due to torch compile
                    if layer_idx in cross_attn_indices or 'lm_head' in name:
                        p.requires_grad = True
            
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"Freeze self attn: {trainable} trainable parameters (cross-attention only)")
    
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )

        scheduler = self._configure_lr_schedulers(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    
    def _configure_lr_schedulers(self, optimizer):
        # Warmup: linear increase from near zero to the base learning rate
        warmup_updates = self.training_config.get("warmup_updates", 1_000)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-12,
            end_factor=1.0,
            total_iters=warmup_updates,
        )

        # Cosine annealing: decay from lr to end_lr over the remaining steps
        total_updates = self.training_config.get("max_steps", 10_000)  # total training steps
        end_lr = self.training_config.get("end_learning_rate", 1e-5)
        decay_iters = max(0, total_updates - warmup_updates)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_iters,
            eta_min=end_lr,
        )

        # Combine warmup and cosine decay sequentially
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_updates],
        )

        return scheduler


    def _configure_fp8_optimizers(self, lr, betas, eps, weight_decay, freeze_attn):
        if freeze_attn:
            self.model.requires_grad_(False)

            cross_attn_indices = [str(i) for i, x in enumerate(self.layer_spec) if x == 'cross']
            for name, p in self.model.named_parameters():
                layer_idx = name.split('.')[2]  # after _orig_mod. due to torch compile
                if layer_idx in cross_attn_indices or 'lm_head' in name:
                    p.requires_grad = True

        hidden_linear_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen layers

            if name in ["fc1", "fc2", 'w12', 'w3']:
                hidden_linear_params.append(params)
            else:
                base_params.append(param)

        d_base = 320  # we search for hyper params from smallest model, then transfer
        d_model = self.model_config.get("embed_dim", 1280)
        hidden_lr_scale = (d_base / d_model) ** 0.5
        
        optim_groups = [
            {
                "params": base_params, 
                "lr": lr
            },
            {
                "params": hidden_linear_params, 
                "lr": lr * hidden_lr_scale
            }
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        return optimizer