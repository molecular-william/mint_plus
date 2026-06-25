import json
import time
from typing import Any, Dict, Optional
import types

import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from galore_torch import GaLoreAdamW 

from mint_plus.utils.log import get_logger
from mint_plus.models.esm2 import MINT
from mint_plus.models import MODEL_REGISTRY

logger = get_logger(__name__)


class MINTWrapper(pl.LightningModule):
    def __init__(
        self,
        model: MINT,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.model_config = model_config
        self.training_config = training_config

        config = MODEL_REGISTRY[self.model_config.get("size", "8M")]
        num_layers=config["num_layers"],

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        loss = self._shared_step(batch)

        # Show global_step in progress bar
        self.manual_backward(loss)

        return None

    def on_train_start(self):
        # Ensure hooks are registered once training actually begins
        register_galore_hooks(self.model, self.optimizer_dict)

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
        freeze_attn = self.training_config.get("freeze_self_attn", False)
        if freeze_attn:
            self.model.requires_grad_(False)
            for name, p in self.model.named_parameters():
                if 'multimer_attn' in name or 'lm' in name or 'norm' in name:
                    p.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Freeze self attn: {trainable} trainable parameters (cross-attention only)")

        # 1. Initialize an optimizer dictionary
        self.optimizer_dict = {}
        
        # 2. Assign one GaLore instance per trainable parameter
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # You can group params by shape if you want to avoid 
                # registering hooks for 1D biases/norms.
                self.optimizer_dict[p] = GaLoreAdamW(
                    [{
                        'params': p, 
                        'rank': self.training_config.get("galore_rank", 128),
                        'update_proj_gap': self.training_config.get("galore_update_proj_gap", 200),
                        'scale': self.training_config.get("galore_scale", 0.25)
                    }],
                    lr=self.training_config.get("lr", 1e-4)
                )
    
        # 3. Return a dummy optimizer to satisfy Lightning's requirements
        # We do NOT return the real optimizers here because we handle them via hooks.
        return torch.optim.SGD([self.model.parameters().__next__()], lr=0)


def register_galore_hooks(model, optimizer_dict):
    for p in model.parameters():
        if p.requires_grad:
            # This hook runs immediately after the backward pass computes grad for p
            def hook(grad):
                if p in optimizer_dict:
                    # Perform the update
                    optimizer_dict[p].step()
                    # Manually clear the gradient to free VRAM immediately
                    p.grad = None 
            
            # Use register_post_accumulate_grad_hook for PyTorch >= 2.1
            p.register_post_accumulate_grad_hook(hook)

# Call this after initializing your model in MINTWrapper or MINTTrainer
# register_galore_hooks(self.model, self.optimizer_dict)