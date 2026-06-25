from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with inheritance support.

    Supports the "extends" key for config inheritance:
        extends: ../base/650M.yaml
        model:
            freeze_self_attn: true

    Args:
        config_path: Path to the YAML config file.
        overrides: Dictionary of override values (e.g., from CLI).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve inheritance
    config = _resolve_inheritance(config, config_path.parent)

    # Apply overrides
    if overrides:
        config = _apply_overrides(config, overrides)

    return config


def _resolve_inheritance(config: Dict, base_dir: Path, visited: Optional[set] = None) -> Dict:
    """
    Resolve config inheritance by merging with parent config.

    Args:
        config: Current config dictionary.
        base_dir: Base directory for resolving relative paths.
        visited: Set of visited paths (to detect cycles).
    """
    if visited is None:
        visited = set()

    extends = config.pop("extends", None)
    if extends is None:
        return config

    # Resolve path
    extends_path = Path(extends)
    if not extends_path.is_absolute():
        extends_path = base_dir / extends_path

    if str(extends_path) in visited:
        raise ValueError(f"Circular dependency detected: {extends_path}")

    visited.add(str(extends_path))

    # Load parent config
    if not extends_path.exists():
        raise FileNotFoundError(f"Parent config not found: {extends_path}")

    with open(extends_path) as f:
        parent_config = yaml.safe_load(f)

    # Recursively resolve parent inheritance
    parent_config = _resolve_inheritance(parent_config, extends_path.parent, visited)

    # Merge: child overrides parent
    merged = _deep_merge(parent_config, config)

    return merged


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_overrides(config: Dict, overrides: Dict) -> Dict:
    """
    Apply flat override dictionary to nested config.
    """
    result = config.copy()
    for key, value in overrides.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def get_model_config(model_size: str) -> Dict:
    """
    Get configuration for a model size.

    Args:
        model_size: Model size identifier (e.g., "8M", "fast_esm_8M", "650M").

    Returns:
        Dictionary with model parameters.

    Raises:
        ValueError: If model_size is not in registry.

    Example:
        >>> config = get_model_config("650M")
        >>> config["num_layers"]
        33
    """
    if model_size not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_size]


def list_available_models() -> Dict[str, Dict]:
    """List all available model sizes and their descriptions."""
    return {
        size: {
            "layers": cfg["num_layers"],
            "embed_dim": cfg["embed_dim"],
            "heads": cfg["attention_heads"],
            "intermediate_size": cfg["intermediate_size"],
            "description": cfg["description"],
        }
        for size, cfg in MODEL_REGISTRY.items()
    }