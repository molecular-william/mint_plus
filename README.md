MINT Plus -- Optimized PPI Training with ESM-2
================================================

A clean, modular reimplementation of MINT (Multimeric Interaction Transformer)
for training protein language models on protein-protein interaction (PPI)
prediction, with performance optimizations including a fused multi-pathway
attention kernel, block-level gradient checkpointing, and LoRA support.

MINT extends ESM-2 with cross-chain attention: each transformer layer has two
attention pathways -- self-attention (with RoPE, intra-chain) and multimer
cross-attention (no RoPE, inter-chain). Their logits are combined before
softmax, producing a single normalized attention distribution over all tokens,
then separated by chain-mask for the weighted sum. This preserves the critical
"combined softmax" semantics from the MINT paper.

Reference: Ullanat, V. et al. "Learning the language of protein-protein
interactions." Nature Communications (2026) 17:1199.
DOI: https://doi.org/10.1038/s41467-025-67971-3

================================================================================
Directory Structure
================================================================================

  mint_plus/
    __init__.py              -- Package metadata, version
    models/
      __init__.py            -- MODEL_REGISTRY (model size configs)
      alphabet.py            -- Token vocabulary (ESM-1b: 33 tokens)
      attention.py           -- MultiHeadAttention, MultimerAttention
      esm2.py                -- MINT model class, weight loading, from_config()
      modules.py             -- TransformerLayer_MINT, VanillaFeedForward,
                                RobertaLMHead, CheckpointedBlock, build_*
      modules_plus.py        -- TransformerLayer_MINT_plus (fused QKV),
                                build_mint_plus()
      rotary_embedding.py    -- RoPE implementation
      kernels/
        __init__.py          -- fused_multimer_combine (Triton)
        multi_pathway_attention.py   -- super-fused attention kernel
    training/
      config.py              -- YAML loader with inheritance
      wrapper.py             -- MINTWrapper (LightningModule)
      trainer.py             -- MINTTrainer (Lightning Trainer builder)
    data/
      data.py                -- STRINGDataset (IterableDataset), CollateFn
    utils/
      log.py                 -- Logging setup

  profiles/
    profile_step.py            -- 150M frozen step profiling
    profile_superfused_step.py -- Profiling with super-fused kernel
    profile_turing.py          -- Turing (2080 Ti) compatibility test

  profile_mint.py              -- Comprehensive 6-phase profiler
  PROFILING_REPORT.md          -- Combined profiling results

  configs/
    base/
      8M.yaml, 35M.yaml, 150M.yaml, 650M.yaml  -- Base model configs
      lora.yaml                                  -- LoRA defaults
      fast_esm_*.yaml                           -- FastESM variants
    recipes/
      frozen_8M.yaml, frozen_35M.yaml, frozen_150M.yaml  -- Freeze mode
      no_frozen_8M.yaml                                  -- Full fine-tune

  mint/ (original MINT reference implementation by authors)
    See mint/README.md for original documentation

================================================================================
Configuration Reference (YAML)
================================================================================

Configs use YAML with inheritance: a recipe YAML extends a base YAML via the
"extends" key. Child keys override parent keys via deep merge.

  extends: ../base/150M.yaml

All config fields are organized under four top-level sections. Below is the
complete reference of every field read by the code.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model: section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

size                        Model size identifier. Must be one of the keys in
                            MODEL_REGISTRY: 8M | 35M | 150M | 650M | 3B | 15B
                            or the fast_esm_ variants.
  Type: string
  Default: "8M"

use_multimer                Enable multimer (cross-chain) attention layers.
                            When false, the model runs as standard ESM-2.
  Type: bool
  Default: true

token_dropout               Apply ESM-2 style token dropout (zero mask token
                            embeddings and rescale by expected masking ratio).
  Type: bool
  Default: true

use_rmsnorm                 Use nn.RMSNorm instead of ESM1bLayerNorm in all
                            transformer layers and final norm.
  Type: bool
  Default: false

use_erf_gelu                Use erf-based GELU (matching the original MINT paper
                            and ESM-2 pretrained weights) instead of PyTorch's
                            default tanh-approximation GELU.
  Type: bool
  Default: false

checkpoint                  Path to a pretrained ESM-2 checkpoint (.bin or .pt).
                            Loaded via load_pretrained_weights(). If the file
                            does not exist, starts with Xavier initialization.
  Type: string or null
  Default: null

  Example: ./ckpts/esm2_150M/pytorch_model.bin

apply_lora                  Apply LoRA (Low-Rank Adaptation) to the model's
                            attention projections. When true, get_peft_model()
                            wraps the base model. See lora_* subfields.
  Type: bool
  Default: false

lora_rank                   LoRA rank (r).
  Type: int
  Default: 8

lora_alpha                  LoRA alpha scaling factor.
  Type: float
  Default: 16.0

lora_dropout                LoRA dropout probability.
  Type: float
  Default: 0.05

lora_target_modules         Which modules to attach LoRA adapters to.
                            Actual code passes ["q_proj", "k_proj", "v_proj"].
  Type: string (currently ignored by code -- uses hardcoded list)

freeze_backbone             Freeze all pretrained backbone parameters when
                            using LoRA. Stored in config but handled by
                            get_peft_model() internally.
  Type: bool
  Default: true

merge_lora                  Merge LoRA weights into base weights after training.
  Type: bool
  Default: false

embed_dim                   Explicit embedding dimension (optional). Can override
                            the registry value for custom-sized experiments.
  Type: int
  Default: auto (from MODEL_REGISTRY[size])

try_flex                    Use the flex attention variant (MINT_flex).
  Type: bool
  Default: false

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training: section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lr                          Peak learning rate. Used by AdamW and as the
                            starting point for LR schedulers.
  Type: float
  Default: 1e-4

max_steps                   Total training steps. Used by LR scheduler to
                            compute decay length and by the Lightning Trainer.
  Type: int
  Default: 500_000 (in wrapper._configure_lr_schedulers, was 10_000 in base)

batch_size                  Batch size (per GPU if DDP).
  Type: int
  Default: 2

accumulate_grad             Gradient accumulation steps. Effective batch =
                            batch_size * accumulate_grad * num_gpus.
  Type: int
  Default: 2

warmup_updates              Number of linear warmup steps. LR goes from near
                            zero to peak over this many steps.
  Type: int
  Default: 2_000 (paper default; was 1_000 in earlier code)

end_learning_rate           Final learning rate after decay. For the linear
                            decay schedule: used as end_factor = end_lr / lr.
                            For cosine: used as eta_min.
  Type: float
  Default: 4e-5 (1/10 of default peak 4e-4)

linear_decay                Use the original MINT paper's LR schedule instead
                            of cosine annealing. The paper schedule:
                              warmup (2,000 steps) -> linear decay to 1/10 peak
                              over 90% of training -> constant at 1/10 peak.
                            When false (default): cosine annealing from peak
                            to end_learning_rate.
  Type: bool
  Default: false

weight_decay                AdamW weight decay.
  Type: float
  Default: 0.01

adam_betas                  Adam betas as a JSON string: "[beta1, beta2]".
  Type: string (JSON array)
  Default: "[0.9, 0.98]"

adam_eps                    Adam epsilon.
  Type: float
  Default: 1e-8

freeze_self_attn            Freeze all model parameters except multimer_attn
                            (and optionally lm_head and layer norms).
                            When true:
                              model.requires_grad_(False)
                              for each param:
                                if 'multimer_attn' or 'lm' or 'norm' in name:
                                  param.requires_grad = True
                            When false: train all parameters.
  Type: bool
  Default: false

grad_clip                   Gradient clipping value. Wired in config but the
                            actual Lightning Trainer does NOT use it (commented
                            out in trainer.py).
  Type: float
  Default: 1.0

val_check_interval          Run validation every N training steps.
  Type: int
  Default: 2_000

checkpoint_block_size       Group N consecutive transformer layers into one
                            gradient checkpoint block. Saves activation memory.
                            Must evenly divide num_layers. 0 = per-layer
                            checkpoint (no grouping).
  Type: int
  Default: 0

use_fused_multi_pathway     Enable the super-fused multi-pathway attention
                            kernel. This replaces the before_softmax + combine
                            pipeline with a single Triton kernel, eliminating
                            (B, H, T, T) logit materialization. Measured 5x
                            speedup on 150M, 13.7x on attention combine alone.
  Type: bool
  Default: false

use_compile                 Enable torch.compile(model, mode='reduce-overhead').
                            May cause CUDA errors when combined with
                            checkpointing + Triton kernel + RoPE + bf16.
  Type: bool
  Default: false

fp8                         Enable fp8-aware optimizer configuration. Not a
                            full fp8 training mode -- just adjusts the optimizer
                            parameter groups with scaled LR for hidden layers.
  Type: bool
  Default: false

use_muon                    Use MuonAdamW optimizer (Newton-Schulz based)
                            instead of AdamW.
  Type: bool
  Default: false

use_galore                  Use GaLoreAdamW optimizer (low-rank projection
                            for memory-efficient training).
  Type: bool
  Default: false

galore_rank                 GaLore projection rank.
  Type: int
  Default: 64

galore_update_proj_gap      Steps between GaLore projection updates.
  Type: int
  Default: 100

galore_scale                GaLore scale factor.
  Type: float
  Default: 0.25

galore_proj_type            GaLore projection type ("std", "reverse_std",
                            "right", "left", "full").
  Type: string
  Default: "std"

use_lora                    Apply LoRA adapters. Set to true and see model:.*
                            lora_* fields.
  Type: bool
  Default: false

print_freq                  How often to print/log training stats (read from
                            config but not actively used by MINT+ trainer).
  Type: int
  Default: 100

checkpoint_every            How often to save model checkpoints.
                            (Read from output: section in trainer, but also
                            available here for backward compat.)
  Type: int
  Default: 2_000

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data: section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_dir                    Directory containing the STRING-DB dataset files
                            (training_filtered.links.txt.gz,
                             training_filtered.seqs.txt.gz,
                             validation.links.txt.gz,
                             validation.seqs.txt.gz).
  Type: string
  Default: "./data"

crop_length                 Truncation length per protein chain. Two chains
                            are concatenated, so total sequence length is
                            typically 2 * crop_length. Random crop preserves
                            <cls> and <eos> tokens at start and end.
  Type: int
  Default: 512

val_examples                Maximum number of validation examples per GPU.
  Type: int
  Default: 250_000

val_max_len                 Maximum validation sequence length (stored in
                            config but not read by the current data pipeline).
  Type: int
  Default: 1024

num_workers                 DataLoader num_workers.
  Type: int
  Default: 4

split                       Dataset split type (e.g., "filtered"). Stored in
                            config for reference, not read by data pipeline.
  Type: string
  Default: "filtered"

max_examples                Maximum training examples per GPU. 0 = unlimited.
  Type: int
  Default: 0

overfit                     Overfit mode: use N batches repeatedly. Stored in
                            config for reference. Not read by current code.
  Type: int
  Default: 0

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
output: section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

run_name                    Run name. Used for checkpoint directory:
                            ./ckpts/{run_name}/ and wandb run name.
  Type: string
  Default: "placeholder"

checkpoint_dir              Base directory for checkpoints.
  Type: string
  Default: "./ckpts"

checkpoint_every            Save checkpoint every N training steps.
  Type: int
  Default: 2_000

save_top_k                  Lightning ModelCheckpoint save_top_k. -1 = keep all.
  Type: int
  Default: -1

wandb                       Enable Weights & Biases logging.
  Type: bool
  Default: false

================================================================================
Usage
================================================================================

Train a model (from project root):

  python -m mint_plus train --config configs/recipes/frozen_150M.yaml

Use programmatically:

  from mint_plus.training.trainer import MINTTrainer
  trainer = MINTTrainer.from_config("configs/recipes/frozen_150M.yaml")
  trainer.fit()

Create a custom config by extending a base:

  # my_experiment.yaml
  extends: ../base/150M.yaml
  model:
    size: 150M
    use_multimer: true
    use_erf_gelu: true       # match original paper's activation
    checkpoint: ./ckpts/esm2_150M/pytorch_model.bin

  training:
    lr: 4e-4
    warmup_updates: 2000
    max_steps: 500000
    linear_decay: true        # original paper LR schedule
    use_fused_multi_pathway: true
    checkpoint_block_size: 3
    batch_size: 64

  data:
    data_dir: ./data/diamond
    crop_length: 512
    num_workers: 8

  output:
    run_name: 150M_paper_config
    checkpoint_every: 2000

================================================================================
Model Sizes (MODEL_REGISTRY)
================================================================================

  Key       Layers  Embed Dim  Heads  Head Dim  FF Dim     Parameters
  ---       ------  ---------  -----  --------  ------     ----------
  8M            6       320      20      16      1,280        8M
  35M          12       480      20      24      1,920       35M
  150M         30       640      20      32      2,560      150M
  650M         33     1,280      20      64      5,120      650M
  3B           36     1,792      16     112      7,168        3B
  15B          48     5,120      20     256     20,480       15B

  FastESM variants (Synthyra HF Hub):
    fast_esm_8M, fast_esm_35M, fast_esm_150M, fast_esm_650M, fast_esm_3B

================================================================================
Key Optimizations
================================================================================

1. Super-fused Multi-Pathway Attention

  A Flash-Attention-2-style Triton kernel that fuses 2x BMM + chain-mask
  combine + online softmax + dropout + 2x weighted sum into a single pass.
  Eliminates (B, H, T, T) logit materialization. K/V reuse factor = QPP=8.

  Measured: 13.7x on attention combine, 5.0x end-to-end on 150M frozen.
  Enable: training.use_fused_multi_pathway: true

2. Block-Level Gradient Checkpointing

  Groups N layers into one checkpoint segment. At N=3 (block-3), saves ~50%
  of activation memory vs per-layer checkpointing. Must evenly divide the
  total layer count (e.g., 30 layers, block_size=3 gives 10 segments).

  Enable: training.checkpoint_block_size: 3

3. Original Paper LR Schedule

  Linear warmup for 2,000 steps, then linear decay to 1/10 peak over 90%
  of total training, then constant. The alternative is cosine annealing.

  Enable: training.linear_decay: true

4. Erf-based GELU

  Matches the original ESM-2/MINT GELU activation. Important for pretrained
  weight compatibility. Difference from default tanh-GELU is ~1e-6 to 1e-4
  per activation.

  Enable: model.use_erf_gelu: true

5. LoRA Fine-Tuning

  Attach low-rank adapters to q_proj, k_proj, v_proj in all attention layers.
  Freezes the backbone. Much lower memory than full fine-tune.

  Enable: training.use_lora: true, model.lora_rank: 16

6. Fused QKV (MultimerAttention class)

  The MultimerAttention class uses a single fused_qkv Linear for all three
  projections. At E >= 1280 (650M+), this is faster than 3 separate GEMMs.
  At E=640 (150M), separate GEMMs are 9% faster.

================================================================================
Performance Benchmarks
================================================================================

Measured on RTX 5000 Ada Generation (33.8 GB VRAM, CC 8.9):
  150M frozen, B=32, T=1024, bf16, block-3 checkpoint:

  Variant                         Step (ms)  tok/s     VRAM (GB)
  ----------------------------------------------------------------
  Baseline (no super-fused)        3,567      9,213     6.39
  Super-fused QPP=1                  993     32,994     2.67
  Super-fused QPP=8                  715     45,816     2.67
  Super-fused + compile            ~600      ~55,000    ~2.7

  With B=80 (projected):          ~285      ~112,000   ~4.5

================================================================================
Turing (RTX 2080 Ti) Compatibility
================================================================================

  GPU: CC 7.5, 11 GB VRAM, no native bf16 tensor cores.

  Super-fused kernel: NOT compatible (Triton LLVM bug on CC 7.5).
  Fallback pipeline (before_softmax + fused_multimer_combine): compatible.
  Must use fp16 precision (not bf16) and use_fused_multi_pathway: false.

  Performance at B=16, T=1024, fp16, 3x Turing DDP: ~14,000 tok/s total.

================================================================================
Data Format
================================================================================

The STRINGDataset reads two gzip files:

  training_filtered.links.txt.gz    -- PPI pairs (tab-separated: name1 name2)
  training_filtered.seqs.txt.gz     -- Sequences (tab-separated: name seq)
  validation.links.txt.gz           -- Same format for validation
  validation.seqs.txt.gz            -- Same format for validation

The CollateFn encodes each chain with:
  "<cls>" + seq.replace("J", "L") + "<eos>"
then concatenates both chains and creates chain_id tensors.

================================================================================
Alphabet (Token Layout)
================================================================================

ESM-1b architecture (33 tokens):

  Index  Token          Description
  -----  -----          -----------
  0      <cls>          Start-of-sequence (BOS) token
  1      <pad>          Padding token
  2      <eos>          End-of-sequence (EOS) token
  3      <unk>          Unknown token
  4-23   L A G V S E R T I D P K Q N F Y M H W C   Standard amino acids
  24     X              Unknown amino acid
  25     B              Asparagine or aspartic acid
  26     U              Selenocysteine
  27     Z              Glutamine or glutamic acid
  28     O              Pyrrolysine
  29     .              Gap character
  30     -              Gap character
  31     <null>         Padding to 8-byte alignment
  32     <mask>         Mask token (for MLM)
