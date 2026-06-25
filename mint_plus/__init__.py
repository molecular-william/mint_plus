"""
MINT Plus: Refactored Protein-Protein Interaction Prediction with ESM-2
=======================================================================

A clean, modular reimplementation of MINT (Modeling Interactions with
Transformer) for training protein language models to predict protein-protein
interactions.

Key features:
- YAML-based configuration system with recipe inheritance
- Model registry for automatic weight loading
- Separate data pipeline with DDP sharding
- Parameter-efficient training modes (freeze, copy, full fine-tune)
- Built-in evaluation for PPI and contact prediction

Usage:
    # Train a model
    python -m mint_plus train --config configs/recipes/frozen_650M.yaml

    # Evaluate a trained model
    python -m mint_plus evaluate --config configs/base/650M.yaml --ckpt path/to/checkpoint.pt

    # Use programmatically
    from mint_plus import MINTTrainer, ModelConfig, TrainingConfig
    trainer = MINTTrainer.from_config("configs/recipes/frozen_650M.yaml")
    trainer.fit()

Reference:
    Rives, A., et al. (2021). "Biological structure and function emerge
    from scaling unsupervised learning to 250 million protein sequences."
    PNAS. https://doi.org/10.1073/pnas.2016239118
"""

__version__ = "0.1.0"
