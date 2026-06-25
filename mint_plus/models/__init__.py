MODEL_REGISTRY = {
    # ESM-2 models (original Fairseq format)
    "8M": {
        "name": "esm2_t6_8M_UR50D",
        "num_layers": 6,
        "embed_dim": 320,
        "attention_heads": 20,
        "intermediate_size": 1280,
        "weight_url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        "description": "Small ESM-2 model for quick experiments",
    },
    "35M": {
        "name": "esm2_t12_35M_UR50D",
        "num_layers": 12,
        "embed_dim": 480,
        "attention_heads": 20,
        "intermediate_size": 1920,
        "weight_url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "description": "Medium-small ESM-2 model for prototyping",
    },
    "150M": {
        "name": "esm2_t30_150M_UR50D",
        "num_layers": 30,
        "embed_dim": 640,
        "attention_heads": 20,
        "intermediate_size": 2560,
        "weight_url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "description": "Medium ESM-2 model for balanced speed/quality",
    },
    "650M": {
        "name": "esm2_t33_650M_UR50D",
        "num_layers": 33,
        "embed_dim": 1280,
        "attention_heads": 20,
        "intermediate_size": 5120,
        "weight_url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "description": "Large ESM-2 model - recommended for production",
    },
    "3B": {
        "name": "esm2_t36_3B_UR50D",
        "num_layers": 36,
        "embed_dim": 1792,
        "attention_heads": 16,
        "intermediate_size": 7168,
        "weight_url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        "description": "Very large ESM-2 model - requires 8x A100",
    },
    "15B": {
        "name": "esm2_t48_15B_UR50D",
        "num_layers": 48,
        "embed_dim": 5120,
        "attention_heads": 20,
        "intermediate_size": 20480,
        "weight_url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
        "description": "Largest ESM-2 model - requires multiple GPUs with high VRAM",
    },
    # FastESM models (Synthyra-compatible)
    "fast_esm_8M": {
        "name": "fast_esm_t6_8M",
        "num_layers": 6,
        "embed_dim": 320,
        "attention_heads": 20,
        "intermediate_size": 1280,
        "weight_url": "Synthyra/ESM2-8M",  # HF Hub repo ID
        "description": "Small FastESM model for quick experiments",
    },
    "fast_esm_35M": {
        "name": "fast_esm_t12_35M",
        "num_layers": 12,
        "embed_dim": 480,
        "attention_heads": 20,
        "intermediate_size": 1920,
        "weight_url": "Synthyra/ESM2-35M",  # HF Hub repo ID
        "description": "Medium-small FastESM model for prototyping",
    },
    "fast_esm_150M": {
        "name": "fast_esm_t30_150M",
        "num_layers": 30,
        "embed_dim": 640,
        "attention_heads": 20,
        "intermediate_size": 2560,
        "weight_url": "Synthyra/ESM2-150M",  # HF Hub repo ID
        "description": "Medium FastESM model for balanced speed/quality",
    },
    "fast_esm_650M": {
        "name": "fast_esm_t33_650M",
        "num_layers": 33,
        "embed_dim": 1280,
        "attention_heads": 20,
        "intermediate_size": 5120,
        "weight_url": "Synthyra/ESM2-650M",  # HF Hub repo ID
        "description": "Large FastESM model - recommended for production",
    },
    "fast_esm_3B": {
        "name": "fast_esm_t36_3B",
        "num_layers": 36,
        "embed_dim": 1792,
        "attention_heads": 16,
        "intermediate_size": 7168,
        "weight_url": "Synthyra/ESM2-3B",  # HF Hub repo ID
        "description": "Very large FastESM model - requires 8x A100",
    },
}