from typing import Union, Dict, Optional, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mint_plus.models.alphabet import Alphabet
from mint_plus.models.modules import TransformerLayer_MINT, RobertaLMHead, ESM1bLayerNorm
#from mint_plus.models.modules_pooled import TransformerLayer_MINT_pooled
from mint_plus.models import MODEL_REGISTRY
from torch.utils.checkpoint import checkpoint


class MINT(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
        use_multimer: bool = True,
        use_rmsnorm: bool = False,
        use_erf_gelu: bool = False,
        fp8: bool = False,  # not implemented yet
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        
        self.token_dropout = token_dropout
        self.use_multimer = use_multimer
        self.use_rmsnorm = use_rmsnorm
        self.use_erf_gelu = use_erf_gelu
        
        self._init_submodules()

    def _init_submodules(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx,
        )
    
        self.layers = nn.ModuleList(
            [
                TransformerLayer_MINT(#_pooled(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    use_rmsnorm=self.use_rmsnorm,
                    use_rotary_embeddings=True,
                    use_multimer=self.use_multimer,
                    use_erf_gelu=self.use_erf_gelu,
                ) for _ in range(self.num_layers)
            ]
        )
            
        self.emb_layer_norm_after = nn.RMSNorm(self.embed_dim) if self.use_rmsnorm else ESM1bLayerNorm(self.embed_dim)
        
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(
        self, 
        tokens: torch.Tensor, 
        chain_ids: Optional[torch.Tensor] = None,
        repr_layers: List[int] = [],
    ) -> Dict[str, torch.Tensor]:

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        if chain_ids is None:
            chain_ids = torch.zeros_like(tokens)
        self_attn_mask = ~torch.eq(chain_ids.unsqueeze(-1), chain_ids.unsqueeze(-2)) # B, T, T
    
        x = self.embed_tokens(tokens)

        # Token dropout implementation
        if self.token_dropout and self.training:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        # NOTE: Always pass padding_mask tensor — removing the data-dependent
        # `if not padding_mask.any(): padding_mask = None` guard avoids a
        # torch._dynamo graph break that causes CUDA stream corruption when
        # combined with per-layer checkpointing + Triton kernel + bf16 compile.
        # An all-False padding_mask is harmless (masked_fill is a no-op).

        # PyTorch checkpointing requires the input tensor to track gradients.
        # Skip during torch.compile -- Dynamo handles differentiation through
        # the compiled graph and requires_grad_() creates a graph break.
        if self.training and not x.requires_grad and not torch.compiler.is_compiling():
            x.requires_grad_(True)
        
        for layer_idx, layer in enumerate(self.layers):
            x, attn = checkpoint(
                layer,
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask,
                chain_ids=chain_ids,
                use_reentrant=False,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
            
        logits = self.lm_head(x)
        result = {"logits": logits, "representations": hidden_representations}

        return result

    @classmethod
    def from_config(cls, model_size: str, fp8: bool = False, use_erf_gelu: bool = False) -> "ESM2":
        """
        Create model from registry config.

        Example:
            >>> model = ESM2.from_config("650M", use_multimer=True)
        """
        model_sizes = ['8M', '35M', '150M', '650M', '3B', '15B']
        if model_size not in model_sizes:
            raise ValueError(
                f"Unknown model size: {model_size}. "
                f"Available: {model_sizes}"
            )

        config = MODEL_REGISTRY[model_size]
        return cls(
            num_layers=config["num_layers"],
            embed_dim=config["embed_dim"],
            attention_heads=config["attention_heads"],
            fp8=fp8,
            use_erf_gelu=use_erf_gelu,
        )

    
    def load_pretrained_weights(
        self, 
        checkpoint_path: str, 
        strict: bool = False, 
        dtype: torch.dtype = torch.float32,
        alternating: bool = False    
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 1. Run the key upgrades (This returns a dictionary with ALL layers mapped!)
        state_dict = self._upgrade_state_dict(state_dict)

        if dtype != torch.float32:  
            state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}

        # 2. Map old fairseq decoder names to lm_head if needed
        if "lm_head.decoder.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["lm_head.decoder.weight"]
        
        # 3. Load the whole state_dict directly!
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        '''
        if missing:
            print(f"Missing keys (expected for criss layers and possibly LM head bias):")
            for k in missing[:20]:
                print(f"   - {k}")
            if len(missing) > 20:
                print(f"   ... and {len(missing)-20} more")
        if unexpected:
            print(f"Unexpected keys (ignored):")
            for k in unexpected[:10]:
                print(f"   - {k}")
        '''


    def _upgrade_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Removes prefixes and ensures the layer names match."""
        upgraded = {}
        for key, value in state_dict.items():
            new_key = key
            
            # Standard prefixes removal
            for p in ["esm.", "encoder."]:
                if new_key.startswith(p):
                    new_key = new_key[len(p):]
            
            parts = new_key.split(".")
            
            # Only try to parse layer_idx if this looks like a layer key
            if len(parts) > 1 and parts[0] in ["layer", "layers"] and parts[1].isdigit():
                layer_idx = parts[1]

                if "attention.self.query" in new_key:
                    new_key = f"layers.{layer_idx}.self_attn.q_proj.{parts[-1]}"
                elif "attention.self.key" in new_key:
                    new_key = f"layers.{layer_idx}.self_attn.k_proj.{parts[-1]}"
                elif "attention.self.value" in new_key:
                    new_key = f"layers.{layer_idx}.self_attn.v_proj.{parts[-1]}"
                elif "attention.output.dense" in new_key:
                    new_key = f"layers.{layer_idx}.self_attn.out_proj.{parts[-1]}"
                elif "attention.Layer" in new_key:
                    new_key = f"layers.{layer_idx}.self_attn_layer_norm.{parts[-1]}"
                elif "LayerNorm" in new_key:
                    new_key = f"layers.{layer_idx}.final_layer_norm.{parts[-1]}"
                elif "intermediate" in new_key:
                    new_key = f"layers.{layer_idx}.feed_forward.fc1.{parts[-1]}"
                elif "output" in new_key:
                    new_key = f"layers.{layer_idx}.feed_forward.fc2.{parts[-1]}"
            else:
                # Handle non-layer root keys
                if "word_embeddings.weight" in new_key:
                    new_key = "embed_tokens.weight"
                elif "lm_head.decoder.weight" in new_key:
                    new_key = "lm_head.weight"
            
            upgraded[new_key] = value
            #print(new_key)
        return upgraded


    def _setup_alphabet(self, alphabet: Union[Alphabet, str]) -> None:
        """Convert alphabet string to Alphabet object and store relevant indices."""
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos