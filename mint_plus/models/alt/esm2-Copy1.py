from typing import Union, Dict, Optional, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from mint_plus.models.alphabet import Alphabet
from mint_plus.models.modules import TransformerLayer_MINT, RobertaLMHead, ESM1bLayerNorm
from mint_plus.models.modules_flex import TransformerLayer_flex
from mint_plus.models.modules_opt import TransformerLayer_Opt, compute_per_chain_positions, CrossAttentionLayer, SelfAttentionLayer
from mint_plus.models.modules_opt import CrossAttentionLayer_fp8, SelfAttentionLayer_fp8
from mint_plus.models import MODEL_REGISTRY


class ESM2(nn.Module):
    def __init__(
        self,
        layer_spec: List[str] = None,  # either self or cross
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        token_dropout: bool = True,
        use_multimer: bool = True,
        try_opt: bool = False,  # try the optimized version, still testing
        alphabet: Union[Alphabet, str] = "ESM-1b",
        try_flex: bool = False,
    ):
        super().__init__()
        if not layer_spec:
            self.layer_spec = ['self'] * num_layers
        else:
            self.layer_spec = layer_spec
            assert layer_spec.count('self') == num_layers, f"layer_spec contains {layer_spec.count('self')} 'intra' layers, but original_num_layers is {num_layers}. They must match."
            
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        
        self._setup_alphabet(alphabet)
        
        self.token_dropout = token_dropout
        self.use_multimer = use_multimer
        self.try_opt = try_opt
        self.try_flex = try_flex
        
        self._init_submodules()

    def _init_submodules(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx,
        )
        if self.try_opt:
            self.layers = nn.ModuleList()
            for layer_type in self.layer_spec:
                if layer_type == 'self':
                    self.layers.append(
                        SelfAttentionLayer(
                            self.embed_dim,
                            4 * self.embed_dim,
                            self.attention_heads,
                            use_rotary_embeddings=True,
                            use_swiglu=False,
                            use_rmsnorm=False,
                        )
                    )
                else:
                    self.layers.append(
                        CrossAttentionLayer(
                            self.embed_dim,
                            4 * self.embed_dim,
                            self.attention_heads,
                            use_rotary_embeddings=True,
                            use_swiglu=True,
                            use_rmsnorm=True,
                        )
                    )

        elif self.try_flex:
            print('flex attention!')
            self.layers = nn.ModuleList(
                [
                    TransformerLayer_flex(
                        self.embed_dim,
                        4 * self.embed_dim,
                        self.attention_heads,
                        use_rotary_embeddings=True,
                        use_multimer=self.use_multimer,
                    ) for _ in range(self.num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    TransformerLayer_MINT(
                        self.embed_dim,
                        4 * self.embed_dim,
                        self.attention_heads,
                        use_rotary_embeddings=True,
                        use_multimer=self.use_multimer,
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
        tokens: torch.Tensor,                     # (B, T)
        chain_ids: Optional[torch.Tensor] = None, # (B, T)
        repr_layers: List[int] = [],              # layer indices to return representations for
        need_head_weights: bool = False,
    ) -> dict:
        """
        Forward pass of ESM2.

        Args:
            tokens: Token indices.
            chain_ids: Optional chain identifiers for each position.
            repr_layers: List of layer indices (0‑based) whose hidden states to return.
                         Layer 0 = embedding output, layer k = after k‑th transformer block.
            need_head_weights: If True, return attention weights for all layers.

        Returns:
            Dictionary containing:
                - "logits": (B, T, vocab_size) next-token predictions.
                - "representations": dict mapping layer index to hidden states (B, T, E).
                - "attentions": (optional) tensor of shape (B, L, H, T, T) if need_head_weights.
        """
        # Input validation
        assert tokens.ndim == 2, "tokens must be 2D (batch, sequence)"

        # 1. Prepare masks
        padding_mask = tokens == self.padding_idx                     # (B, T)
        chain_mask = self._build_chain_mask(tokens, chain_ids)       # (B, T, T) or None

        per_chain_positions = None
        if chain_ids is not None and self.use_multimer:
            # For rotary we want positions starting at 0 for each chain separately
            per_chain_positions = compute_per_chain_positions(chain_ids)   # (B, T)

        # 2. Embeddings
        hidden_states = self.embed_tokens(tokens)                    # (B, T, E)
        hidden_states = self._apply_token_dropout(hidden_states, tokens, padding_mask)

        # Zero out padding positions
        hidden_states = hidden_states.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )

        # 3. Store initial representation if requested
        repr_set = set(repr_layers)
        representations = {}
        if 0 in repr_set:
            representations[0] = hidden_states

        # 4. Transformer layers (sequence-first layout)
        hidden_states = hidden_states.transpose(0, 1)                # (T, B, E)
        padding_mask_for_layers = None if not padding_mask.any() else padding_mask

        hidden_states, attn_weights = self._run_transformer_layers(
            hidden_states=hidden_states,
            padding_mask=padding_mask_for_layers,
            chain_mask=chain_mask,
            need_head_weights=need_head_weights,
            repr_layers=repr_set,
            representations=representations,
            position_ids=per_chain_positions,
        )

        # 5. Final layer norm and convert back to batch-first
        hidden_states = self.emb_layer_norm_after(hidden_states)     # (T, B, E)
        hidden_states = hidden_states.transpose(0, 1)                # (B, T, E)

        # 6. Store last representation if requested (already done inside loop for layer == num_layers)
        #    but we need to ensure it's the post-layer-norm version.
        if self.num_layers in repr_set:
            representations[self.config.num_layers] = hidden_states

        # 7. Language modelling head
        logits = self.lm_head(hidden_states)                         # (B, T, vocab_size)

        # 8. Build output dictionary
        result = {
            "logits": logits,
            "representations": representations,
        }
        if need_head_weights:
            attentions = self._postprocess_attentions(attn_weights, padding_mask)
            result["attentions"] = attentions

        return result

        
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
        if not self.use_multimer:
            return None
        if chain_ids is None:
            chain_ids = torch.zeros_like(tokens)
        # Different chain -> mask out attention
        return ~torch.eq(chain_ids.unsqueeze(-1), chain_ids.unsqueeze(-2))

    def _run_transformer_layers(
        self,
        hidden_states: torch.Tensor,          # (T, B, E)
        padding_mask: Optional[torch.Tensor], # (B, T) or None
        chain_mask: Optional[torch.Tensor],   # (B, T, T) or None
        need_head_weights: bool,
        repr_layers: Set[int],
        representations: dict,
        position_ids,
    ):
        """
        Pass hidden states through all transformer layers.

        Returns:
            hidden_states: after all layers (T, B, E)
            attn_weights: list of attention tensors if need_head_weights else None
        """
        attn_weights = [] if need_head_weights else None

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, attn = layer(
                hidden_states,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
                self_attn_mask=chain_mask,
                position_ids=position_ids,
            )

            # Store representation after this layer if requested (1-indexed in paper)
            if layer_idx + 1 in repr_layers:
                # Convert back to (B, T, E) for storage
                representations[layer_idx + 1] = hidden_states.transpose(0, 1)

            # Collect attention weights (convert from (H, B, T, T) to (B, H, T, T))
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))

        return hidden_states, attn_weights

    def _postprocess_attentions(
        self,
        attn_weights: List[torch.Tensor],
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stack layer attentions and mask out padding‑padding positions.

        Returns tensor of shape (B, num_layers, H, T, T).
        """
        # attentions: list of (B, H, T, T) per layer -> stack to (B, L, H, T, T)
        attentions = torch.stack(attn_weights, dim=1)

        # Zero out attention where either query or key is padding
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.float()
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            # Expand mask to (B, 1, 1, T, T) then broadcast
            attentions = attentions * attention_mask[:, None, None, :, :]

        return attentions

        
    @classmethod
    def from_config(cls, model_size: str, use_multimer: bool = True, try_opt: bool = False, layer_spec = None, try_flex: bool = False) -> "ESM2":
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
            use_multimer=use_multimer,
            try_opt=try_opt,
            layer_spec=layer_spec,
            try_flex=try_flex
        )

    
    def load_pretrained_weights(
        self, 
        checkpoint_path: str, 
        strict: bool = False, 
        dtype: torch.dtype = torch.float32,
        alternating: bool = False    
    ):
        """
        Load pretrained ESM-2 weights and cast to the specified dtype.
        """
        # Load directly to CPU first to avoid GPU memory spikes
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove prefixes from fairseq checkpoints
        state_dict = self._upgrade_state_dict(state_dict)

        if dtype != torch.float32:  # cast to desired dtype
            state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}

        new_state = {}

        if "embed_tokens.weight" in state_dict:
            new_state["embed_tokens.weight"] = state_dict["embed_tokens.weight"]

        if "emb_layer_norm_after.weight" in state_dict:
            new_state["emb_layer_norm_after.weight"] = state_dict["emb_layer_norm_after.weight"]

        if "lm_head.weight" in state_dict:
            new_state["lm_head.weight"] = state_dict["lm_head.weight"]
        else:
            # In fairseq checkpoints, decoder.weight might be used
            if "lm_head.decoder.weight" in state_dict:
                new_state["lm_head.weight"] = state_dict["lm_head.decoder.weight"]
        
        self_layer_counter = 0
        for i, layer_type in enumerate(self.layer_spec):
            if layer_type == 'self':
                our_prefix = f"layers.{i}."
                ckpt_prefix = f"layers.{self_layer_counter}."
                for ckpt_key, value in state_dict.items():
                    if ckpt_key.startswith(ckpt_prefix):
                        # Remove original prefix and add our prefix
                        suffix = ckpt_key[len(ckpt_prefix):]
                        our_key = our_prefix + suffix
                        new_state[our_key] = value
                self_layer_counter += 1

        for key in ["embed_tokens.weight", "emb_layer_norm_after.weight", "emb_layer_norm_after.bias"]:
            if key in state_dict:  # if exist
                new_state[key] = state_dict[key]
        
        
        missing, unexpected = self.load_state_dict(new_state, strict=False)

        print(f"Loaded pretrained weights into {self_layer_counter} self attn layers.")
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


    def _upgrade_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        removes prefixes and ensure the layer names match
        """
        upgraded = {}
        for key, value in state_dict.items():
            new_key = key
            # standard prefixes remove
            for p in ["esm.", "encoder."]:
                if new_key.startswith(p):
                    new_key = new_key[len(p):]
            
            parts = new_key.split(".")
            layer_idx = parts[1]  # layer.0.attention.self.query.weight


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
            elif "word_embeddings.weight" in new_key:
                new_key = "embed_tokens.weight"
            elif "lm_head.decoder.weight" in new_key:
                new_key = "lm_head.weight"
            elif "intermediate" in new_key:
                new_key = f"layers.{layer_idx}.feed_forward.fc1.{parts[-1]}"
            elif "output" in new_key:
                new_key = f"layers.{layer_idx}.feed_forward.fc2.{parts[-1]}"
            
            upgraded[new_key] = value
            
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

        