import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, get_peft_model
from typing import Any, Dict, Optional
from pathlib import Path
# from lightning.pytorch.strategies import DDPStrategy

from mint_plus.utils.log import get_logger
from mint_plus.data.data import STRINGDataset, CollateFn
from mint_plus.models.esm2 import MINT
#from mint_plus.models.esm2_flex import MINT_flex
from mint_plus.models.modules import build_checkpointed_model
from mint_plus.models.modules import enable_fused_multi_pathway
from mint_plus.training.wrapper import MINTWrapper
from mint_plus.training.config import load_config

# note that the MLM loss calculation is specified in the model forward

logger = get_logger(__name__)
torch.set_float32_matmul_precision('medium')  # medium

# use config yaml files
class MINTTrainer:
    """
    Example:
        >>> trainer = MINTTrainer.from_config("configs/recipes/frozen_650M.yaml")
        >>> trainer.fit()
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.data_config = config.get("data", {})
        self.output_config = config.get("output", {})

        self.model = self._build_model()
        # Optional: torch.compile (disabled by default -- super-fused kernel
        # already runs at 715 ms/step eager. Compile can OOM or produce
        # illegal-memory-access errors on the RoPE path when combined with
        # checkpointing + Triton kernels. Enable only if eager is too slow.)
        if self.training_config.get("use_compile", False):
            import torch._inductor.config as inductor_cfg
            inductor_cfg.max_fusion_size = 8
            self.model = torch.compile(self.model, mode='reduce-overhead')
            logger.info("torch.compile enabled (reduce-overhead mode)")
        self.wrapper = self._build_wrapper()  # freeze_self_attn is processed here
        self.lightning_trainer = self._build_trainer()
        self.train_loader, self.val_loader = self._build_dataloaders()

    def _build_trainer(self):
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        # Check for interactive/notebook environment
        self.strategy = self._get_training_strategy()
            
        self.max_steps = self.training_config.get("max_steps", 500_000)
        self.accu_grad = self.training_config.get("accumulate_grad", 2)

        run_name = self.output_config.get("run_name", 'placeholder')
        checkpoint_interval = self.training_config.get("checkpoint_every", 2_000)
        val_check_interval = self.training_config.get("val_check_interval", 2_000)
        
        trainer = pl.Trainer(
            default_root_dir=f"./ckpts/{run_name}",
            accelerator=accelerator,
            devices=torch.cuda.device_count(),
            max_steps=self.max_steps,
            num_sanity_val_steps=2,
            enable_progress_bar=True,
           # gradient_clip_val=self.training_config.get("grad_clip", 1.0),  # AdamW fused doesn't need this
            enable_checkpointing=True,
            callbacks=[ModelCheckpoint(dirpath=f"./ckpts/{run_name}", every_n_train_steps=checkpoint_interval),],
            accumulate_grad_batches=self.accu_grad,
            val_check_interval=val_check_interval,
            strategy=self.strategy,
            precision='bf16-mixed' #"16-mixed"
        )
        return trainer

    def _build_dataloaders(self):
        data_dir = self.data_config.get("data_dir", "./data")
        num_workers = self.data_config.get('num_workers', 4)
        collate_fn = CollateFn(self.data_config.get('crop_length', 512))
        self.batch_size = self.training_config.get('batch_size', 2)
            
        val_ds = STRINGDataset(
            links_path=data_dir + '/' + "validation.links.txt.gz",
            seqs_path=data_dir + '/' + "validation.seqs.txt.gz",
            global_rank=self.lightning_trainer.global_rank,
            world_size=self.lightning_trainer.world_size,
            max_examples=self.data_config.get('val_examples', 250_000),
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds, 
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=2
        )

        train_ds = STRINGDataset(
            links_path=data_dir + '/' + "training_filtered.links.txt.gz",
            seqs_path=data_dir + '/' + "training_filtered.seqs.txt.gz",
            global_rank=self.lightning_trainer.global_rank,
            world_size=self.lightning_trainer.world_size,
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2
        )
        return train_loader, val_loader

    def _build_model(self):
        self.model_size = self.model_config.get("size", "8M")
        self.use_multimer = self.model_config.get("use_multimer", True)
        self.try_flex = self.model_config.get("try_flex", False)
        self.fp8 = self.training_config.get("fp8", False)
        self.use_erf_gelu = self.model_config.get("use_erf_gelu", False)
        # creates the backbone
        if self.try_flex:
            model = MINT_flex.from_config(self.model_size, try_flex=True, fp8=self.fp8)
        else:
            model = MINT.from_config(self.model_size, fp8=self.fp8, use_erf_gelu=self.use_erf_gelu)

        dtype = torch.bfloat16#float32#torch.bfloat16
        #model.to(dtype)
        
        ckpt_path = self.model_config.get("checkpoint", None)
        if ckpt_path and Path(ckpt_path).exists():
            # self._weight_sanity_check(model, '')
            model.load_pretrained_weights(ckpt_path, dtype=dtype)
            # self._weight_sanity_check(model, '(pre-training weights) ')
            
            logger.info(f"Starting with pre-trained weights.")
        else:
            logger.info(f"Starting with Xavier initialized weights.")

        if self.training_config.get("use_lora", False):
            peft_config = LoraConfig(
                r=self.training_config.get("lora_r", 8),
                lora_alpha=self.training_config.get("lora_alpha", 16),
                lora_dropout=0.05,
                bias="none",
                # target_modules strings match the specific sub-modules inside your MultiHeadAttention
                target_modules=["q_proj", "k_proj", "v_proj"], 
            )
            # get_peft_model handles the frozen base weights and configures gradients automatically
            model = get_peft_model(model, peft_config)
        
            # Optional: verify parameters
            model.print_trainable_parameters()

        # Apply optimizations: fused multimer attention + block-level checkpointing
        ckpt_block = self.training_config.get("checkpoint_block_size", 0)
        if ckpt_block > 0:
            # flex model -- only block checkpointing, no fused kernel
            model = build_checkpointed_model(model, block_size=ckpt_block)
            logger.info(f"Optimization: block-level checkpointing (block_size={ckpt_block}, flex mode)")

        # Multi-pathway fused attention (Phase 2): avoids (B, H, T, T) logit materialization
        if self.training_config.get("use_fused_multi_pathway", False):
            enable_fused_multi_pathway(model, enabled=True)
            logger.info("Optimization: multi-pathway fused attention kernel enabled (~5x combine speedup)")

        return model

    def _build_wrapper(self):
        return MINTWrapper(
            model = self.model,
            model_config=self.model_config,
            training_config=self.training_config,
        )

    def fit(self):
        """Run the full training loop."""
        logger.info(f"Starting training...")
        logger.info(f"  Model size: {self.model_size}")
        logger.info(f"  Use multimer: {self.use_multimer}")
        logger.info(f"  Freeze self-attn: {self.training_config.get('freeze_self_attn', False)}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Accumulate grad: {self.accu_grad}")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Strategy: {self.strategy}")
        ckpt_block = self.training_config.get("checkpoint_block_size", 0)
        logger.info(f"  Checkpoint blocks: {ckpt_block if ckpt_block > 0 else 'per-layer'}")
        logger.info(f"  Fused attention: {not self.try_flex}")

        self.lightning_trainer.fit(
            self.wrapper,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )
        logger.info("Training complete!")
        # Merge weights and save on the main process
        if self.lightning_trainer.global_rank == 0:
            run_name = self.output_config.get("run_name", 'placeholder')
            save_dir = f"./ckpts/{run_name}/merged_model"
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Extract the raw model from the Lightning Wrapper
            raw_model = self.wrapper.model
            
            # Unwrap torch.compile if it was applied
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
                
            if self.training_config.get("use_lora", False):
                logger.info("Merging LoRA adapters into the base model weights...")
                
                # merge_and_unload combines W_base + (A * B) into a new base weight matrix
                # and removes all PEFT/LoRA specific hook layers
                merged_model = raw_model.merge_and_unload()
                
                logger.info(f"Saving fully merged standalone model to {save_dir}...")
                torch.save(merged_model.state_dict(), f"{save_dir}/merged_model.pt")
                logger.info("Merged model saved successfully!")
            else:
                logger.info(f"LoRA not used. Saving standard model state dict to {save_dir}...")
                torch.save(raw_model.state_dict(), f"{save_dir}/model_weights.pt")

    
    def evaluate(self, ckpt_path: Optional[str] = None):
        """
        Evaluate the model on validation set.

        Args:
            ckpt_path: Path to checkpoint to load. If None, uses current model.
        """
        
        if ckpt_path:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.wrapper.load_state_dict(checkpoint["state_dict"])

        logger.info("Running evaluation...")
        results = self.lightning_trainer.validate(
            self.wrapper,
            dataloaders=self.val_loader,
        )

        logger.info(f"Validation results: {results}")
        return results

    
    @classmethod
    def from_config(cls, config_path: str, overrides: Optional[Dict] = None) -> "MINTTrainer":
        config = load_config(config_path, overrides)
        return cls(config)

    
    def _get_training_strategy(self):
        is_notebook = False
        gpus = list(range(torch.cuda.device_count()))
        
        try:
            # Check if running inside an IPython kernel
            from IPython import get_ipython
            if get_ipython() is not None:
                is_notebook = True
        except ImportError:
            pass

        freeze_attn = self.model_config.get("freeze_self_attn", False)
        
        if is_notebook:
            strategy = "ddp_notebook_find_unused_parameters_true" if freeze_attn else "ddp_notebook"
            logger.info(f"Interactive notebook detected. Using strategy: {strategy}")
        else:
            strategy = "ddp_find_unused_parameters_true" if freeze_attn else "ddp"
            logger.info(f"Standard script environment detected. Using strategy: {strategy}")

        if len(gpus) < 2:
            strategy = "auto"
            logger.info(f"Auto-resolving training strategy.")

        return strategy

        
    def _weight_sanity_check(self, model, step):
        weight_tensor = model.layers[0].self_attn.q_proj.weight.data

        print(f"--- Weight Sanity Check {step}---")
        print(f"Mean of weights: {weight_tensor.mean().item():.6f}")
        print(f"Std  of weights: {weight_tensor.std().item():.6f}")
        print(f"First 5 elements: {weight_tensor[0, :5].tolist()}")