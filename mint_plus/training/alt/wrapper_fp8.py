import json
import time
from typing import Any, Dict, Optional
import types

import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Added TransformerEngine Context Managers
import transformer_engine.pytorch as te
import bitsandbytes as bnb
from transformer_engine.common.recipe import Format, DelayedScaling

from mint_plus.utils.log import get_logger
from mint_plus.models.esm2 import MINT
from mint_plus.models import MODEL_REGISTRY

logger = get_logger(__name__)


class MINTWrapper(pl.LightningModule):
    def __init__(self,
        model: MINT,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.model_config = model_config
        self.training_config = training_config

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        loss = self._shared_step(batch)
        self.log("global_step", self.global_step, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        self.stage = "val"
        loss = self._shared_step(batch)
        return loss

    def _shared_step(self, batch):
        """
        Refactored Forward logic wrapped inside dynamic FP8 automatic scaling.
        """
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
        
        
        # Configure the delayed scaling recipe to observe past max ranges over a step window
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,      # E4M3 for forward, E5M2 for backward
            amax_history_len=1024,         # Range window tracker size
            amax_compute_algo="max"        # Scale targeting the peak observed historical values
        )

        # Execute operations wrapped cleanly within the scaling tracker
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = self.model(inp, chain_ids=chain_ids)["logits"]
                
        # Compute cross-entropy loss averaged over masked positions
        loss = F.cross_entropy(out.transpose(1, 2), tokens, reduction="none")
        loss = (loss * mask).sum() / mask.sum()

        # Log metrics - automatically aggregated across DDP ranks
        self.log(f"{self.stage}_loss", loss, sync_dist=True, prog_bar=True)
        self.log(f"{self.stage}_perplexity", torch.exp(loss), sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # 1. Gather configuration parameters
        lr = self.training_config.get("lr", 1e-4)
        betas = json.loads(self.training_config.get("adam_betas", "[0.9, 0.98]"))
        eps = self.training_config.get("adam_eps", 1e-8)
        weight_decay = self.training_config.get("weight_decay", 0.01)
        freeze_attn = self.training_config.get("freeze_self_attn", False)
        
        # Handle layer freezing configurations if requested
        if freeze_attn:
            self.model.requires_grad_(False)
            for name, p in self.model.named_parameters():
                if 'multimer_attn' in name or 'lm' in name or 'norm' in name:
                    p.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Freeze self attn: {trainable} trainable parameters (cross-attention only)")

        logger.info('Using default AdamW optimizer configuration')
        optimizer = bnb.optim.Adam8bit(#torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
    #        fused=True,
        )

        # 3. Attach Learning Rate Scheduler
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