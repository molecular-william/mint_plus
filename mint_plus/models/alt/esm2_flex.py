from typing import Union, Dict, Optional, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mint_plus.models.alphabet import Alphabet
from mint_plus.models.modules import TransformerLayer_MINT, RobertaLMHead, ESM1bLayerNorm
#from mint_plus.models.modules_flex import TransformerLayer_flex, RobertaLMHead, ESM1bLayerNorm, make_multimer_masks
#from mint_plus.models.modules_opt import TransformerLayer_Opt, compute_per_chain_positions, CrossAttentionLayer, SelfAttentionLayer
#from mint_plus.models.modules_opt import CrossAttentionLayer_fp8, SelfAttentionLayer_fp8
from mint_plus.models import MODEL_REGISTRY


class MINT_flex(nn.Module):
    def __init__(
        self,
        layer_spec: List[str] = None,  # either self or cross
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        token_dropout: bool = True,
        alphabet: Union[Alphabet, str] = "ESM-1b",
        fp8: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        
        self._setup_alphabet(alphabet)
        
        self.token_dropout = token_dropout
        self.try_flex = try_flex
        self.fp8 = fp8
        
        self._init_submodules()

    def _init_submodules(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx,
        )
    
        print('flex attention!')
        self.layers = nn.ModuleList(
            [
                TransformerLayer_flex(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    use_rotary_embeddings=True,
                    fp8=self.fp8
                ) for _ in range(self.num_layers)
            ]
        )
        
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,  # a probability value for each possible token?
            weight=self.embed_tokens.weight,
        )

    def forward(
        self, 
        tokens: torch.Tensor, 
        padding_mask: Optional[torch.Tensor] = None,
        chain_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, current_seq_len = tokens.shape # <-- Extracts exact size (e.g., 2 or 2751)
        hidden_states = self.embed_tokens(tokens)
    
        if padding_mask is None:
            padding_mask = tokens.eq(self.padding_idx)
    
        # Token dropout implementation
        if self.token_dropout and self.training:
            mask = tokens.eq(self.mask_idx)
            
            # If mask.sum() > 0, has_mask is 1.0. If 0, has_mask is 0.0.
            has_mask = (mask.sum() > 0).float()
            mask_ratio = 0.15 * has_mask
            
            p = torch.rand_like(hidden_states[:, :, :1])
            dropout_mask = (p < mask_ratio) & (~padding_mask.unsqueeze(-1))
            hidden_states = torch.where(dropout_mask, 0.0, hidden_states)
    
        # COMPILER SAFE: Zero Graph-Break Mask Routing matching the dynamic length
        if chain_ids is not None:
            # Pass current_seq_len explicitly to build a perfectly sized block_mask
            intra_block_mask, inter_block_mask = make_multimer_masks(chain_ids, current_seq_len)
        else:
            intra_block_mask, inter_block_mask = None, None
    
        # Execute Layer stack cleanly
        hidden_states, _ = self._run_transformer_layers(
            hidden_states,
            intra_block_mask=intra_block_mask,
            inter_block_mask=inter_block_mask,
        )
        
        hidden_states = self.emb_layer_norm_after(hidden_states)
        logits = self.lm_head(hidden_states)
    
        return {"logits": logits, "representations": hidden_states}

        
    def _apply_token_dropout(self, embeddings, tokens, padding_mask):  # huh?
        """
        Apply token dropout regularisation (masked token zeroing + scaling).
        This replicates the BERT training trick where a fraction of masked positions
        are set to zero and the remaining embeddings are scaled to keep expected sum.
        """
        if not self.token_dropout:
            return embeddings

        # Zero out embeddings at mask positions
        mask_positions = (tokens == self.mask_idx).unsqueeze(-1)
        embeddings = embeddings.masked_fill(mask_positions, 0.0)

        # Scaling factor: (1 - mask_ratio_train) / (1 - observed_mask_ratio)
        mask_ratio_train = 0.15 * 0.8  # 12% of tokens are masked in BERT style
        src_lengths = (~padding_mask).sum(-1).to(embeddings.dtype)
        observed_mask_ratio = (tokens == self.mask_idx).sum(-1).to(embeddings.dtype) / src_lengths
        scale = (1 - mask_ratio_train) / (1 - observed_mask_ratio).clamp(min=1e-8)
        embeddings = embeddings * scale[:, None, None]
        return embeddings

    def _build_chain_mask(
        self,
        tokens: torch.Tensor,
        chain_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build attention mask to prevent cross‑chain attention when use_multimer=True.
        Returns a boolean mask of shape (B, T, T) where True means "do not attend"
        (positions belong to different chains). If use_multimer=False, returns None.
        """
        if chain_ids is None:
            chain_ids = torch.zeros_like(tokens)
        # Different chain -> mask out attention
        return ~torch.eq(chain_ids.unsqueeze(-1), chain_ids.unsqueeze(-2))
    '''
    def _run_transformer_layers(
        self,
        hidden_states: torch.Tensor,
        intra_block_mask=None,
        inter_block_mask=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                intra_block_mask=intra_block_mask,
                inter_block_mask=inter_block_mask,
            )
            
        return hidden_states, []
    '''
    def _run_transformer_layers(  # implemented activation checkpointing
        self,
        hidden_states: torch.Tensor,
        intra_block_mask=None,
        inter_block_mask=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        # PyTorch checkpointing requires the input tensor to have requires_grad=True.
        # This ensures the autograd engine tracks the backward pass properly.
        if self.training and not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        for layer in self.layers:
            if self.training:
                # Execute the layer via checkpointing to save activation memory.
                # `use_reentrant=False` is highly recommended for modern PyTorch / Lightning compatibility.
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    use_reentrant=False,
                    intra_block_mask=intra_block_mask,
                    inter_block_mask=inter_block_mask,
                )
            else:
                # Standard execution for validation/inference
                hidden_states = layer(
                    hidden_states,
                    intra_block_mask=intra_block_mask,
                    inter_block_mask=inter_block_mask,
                )
            
        return hidden_states, []
        
    @classmethod
    def from_config(cls, model_size: str, try_flex: bool = False, fp8: bool = False) -> "ESM2":
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
            try_flex=try_flex,
            fp8=fp8,
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

