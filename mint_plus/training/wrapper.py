import json
import time
from typing import Any, Dict, Optional
import types

import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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
        # 1. Gather configuration parameters
        lr = self.training_config.get("lr", 1e-4)
        betas = json.loads(self.training_config.get("adam_betas", "[0.9, 0.98]"))
        eps = self.training_config.get("adam_eps", 1e-8)
        weight_decay = self.training_config.get("weight_decay", 0.01)
        freeze_attn = self.training_config.get("freeze_self_attn", False)
        architecture = self.model_config.get("architecture", "standard")

        # muS: different LRs for different parameter groups
        if architecture == "mus":
            return self._configure_mu_optimizers(lr, betas, eps, weight_decay)

        # Handle layer freezing configurations if requested
        if freeze_attn:
            self.model.requires_grad_(False)
            for name, p in self.model.named_parameters():
                if 'multimer_attn' in name or 'lm' in name or 'norm' in name:
                    p.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Freeze self attn: {trainable} trainable parameters (cross-attention only)")

        # 2. Check if we should use Muon (e.g., via config flag or default fallback)
        if self.training_config.get('use_muon', False):
            optimizer = self._configure_MuonAdamW_optimizers(lr, betas, eps, weight_decay)

        elif self.training_config.get('use_galore', False):
            optimizer = self._configure_GaLore_optimizers(lr, betas, eps, weight_decay)

        elif self.training_config.get('fp8', False):
            logger.info('Using fp8 optimizer configuration')
            optimizer = self._configure_fp8_optimizers(lr, betas, eps, weight_decay)

        else:
            logger.info('Using default AdamW optimizer configuration')
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                fused=True,
            )

        # 3. Attach Learning Rate Scheduler
        scheduler = self._configure_lr_schedulers(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ---- muS Optimizer (different LR per parameter group) ----

    def _configure_mu_optimizers(self, lr, betas, eps, weight_decay):
        """muS optimizer with per-group LR scaling.

        Hidden layers: LR = base_lr * sqrt(d_base / d_new)
        Embedding:     LR = base_lr
        LM head:       LR = base_lr
        """
        from mint_plus.models.mu_scaling.optim import build_mu_param_groups

        hidden_dim = self.model_config.get("embed_dim", 640)
        base_width = self.training_config.get("base_width", 320)

        param_groups = build_mu_param_groups(
            self.model, base_lr=lr, hidden_dim=hidden_dim,
            weight_decay=weight_decay, base_width=base_width,
        )
        logger.info(
            f"muS optimizer: {len(param_groups)} groups, "
            f"LRs={[g['lr'] for g in param_groups]}"
        )

        optimizer = torch.optim.AdamW(
            param_groups, betas=betas, eps=eps, fused=True,
        )

        scheduler = self._configure_lr_schedulers(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    
    def _configure_lr_schedulers(self, optimizer):
        # Warmup: linear increase from near zero to the base learning rate
        warmup_updates = self.training_config.get("warmup_updates", 2_000)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-12,
            end_factor=1.0,
            total_iters=warmup_updates,
        )

        total_updates = self.training_config.get("max_steps", 500_000)
        end_lr = self.training_config.get("end_learning_rate", 4e-5)
        use_linear_decay = self.training_config.get("linear_decay", False)

        if use_linear_decay:
            # Original paper schedule:
            #   warmup over first 2,000 steps to peak 4e-4,
            #   then linear decay to 1/10 peak over 90% of training
            peak_lr = self.training_config.get("lr", 4e-4)
            decay_iters = int(0.9 * total_updates)
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_lr / peak_lr,
                total_iters=decay_iters,
            )
        else:
            # Cosine annealing: decay from lr to end_lr over remaining steps
            decay_iters = max(0, total_updates - warmup_updates)
            decay_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=decay_iters,
                eta_min=end_lr,
            )

        # Combine warmup and decay sequentially
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_updates],
        )

        return scheduler


    def _configure_fp8_optimizers(self, lr, betas, eps, weight_decay):
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


    def _configure_MuonAdamW_optimizers(self, lr, betas, eps, weight_decay):  # seems to make training worse, at least for < 1B
        from mint_plus.training.optim_nanochat import MuonAdamW
        logger.info('Using single-GPU MuonAdamW optimizer configuration')
    
        muon_params = []
        adamw_params = []
    
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
    
        param_groups = [
            {
                "params": muon_params,
                "kind": "muon",
                "lr": lr * 0.2, 
                "momentum": 0.95,
                "ns_steps": 5,
                "beta2": 0.95,
                "weight_decay": weight_decay
            },
            {
                "params": adamw_params,
                "kind": "adamw",
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay
            }
        ]
        optimizer = MuonAdamW(param_groups)
    
        # --- THE FIX FOR PYTORCH LIGHTNING ---
        # Save the original step method
        original_step = optimizer.step

        # Redefine step to accept closure and ignore it (or execute it if required)
        def lightning_compatible_step(self, *args, **kwargs):
            closure = kwargs.pop('closure', None)  # Lightning passes closure via kwargs, scheduler via args[0]
            if not closure and len(args) > 0:
                closure = args[0]
            
            if closure is not None:
                closure()  # Run the objective function evaluation/backward pass if required
            
            return original_step()
        # Correctly bind the function as an instance method of the optimizer
        optimizer.step = types.MethodType(lightning_compatible_step, optimizer)


    def _configure_GaLore_optimizers(self, lr, betas, eps, weight_decay):
        from galore_torch import GaLoreAdamW  # or GaLoreAdamW8bit if using 8-bit quantization
        logger.info('Using GaLoreAdamW optimizer configuration')
    
        # Extract GaLore-specific configurations with defaults
        galore_rank = self.training_config.get("galore_rank", 64)
        update_proj_gap = self.training_config.get("galore_update_proj_gap", 100)
        galore_scale = self.training_config.get("galore_scale", 0.25)
        proj_type = self.training_config.get("galore_proj_type", "std")
    
        galore_params = []
        non_galore_params = []
    
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            
            # GaLore expects 2D weight matrices (e.g., nn.Linear weights)
            # Typically, embedding layers and LM heads are excluded from low-rank projection
            if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name:
                galore_params.append(p)
            else:
                non_galore_params.append(p)
    
        param_groups = [
            {
                "params": non_galore_params,
                "weight_decay": weight_decay
            },
            {
                "params": galore_params,
                "rank": galore_rank,
                "update_proj_gap": update_proj_gap,
                "scale": galore_scale,
                "proj_type": proj_type,
                "weight_decay": weight_decay
            }
        ]
    
        optimizer = GaLoreAdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps
        )
    
        # PyTorch Lightning compatibility step injection:
        # Standard GaLore implementations accept standard `closure` arguments natively,
        # but to explicitly prevent unexpected keyword arguments or edge-case scheduler errors 
        # from Lightning's optimizer loop wrapper, you can bind a safe step wrapper.
        original_step = optimizer.step
    
        def lightning_compatible_step(self, *args, **kwargs):
            closure = kwargs.pop('closure', None)
            if not closure and len(args) > 0:
                closure = args[0]
            
            # Run the closure calculation if provided by Lightning
            loss = None
            if closure is not None:
                loss = closure()
            
            original_step()
            return loss
    
        optimizer.step = types.MethodType(lightning_compatible_step, optimizer)
        return optimizer