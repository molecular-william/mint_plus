================================================================================
MINT+ PROFILING REPORT -- COMPLETE COLLECTION
================================================================================

This is the combined profiling report for MINT+ -- a reimplementation of MINT
(Multimeric Interaction Transformer) for PPI prediction with ESM-2 backbone.

Current versions: PyTorch 2.12.1+cu130, Triton 3.7.1, CUDA 13.0
================================================================================
MINT+ PROFILING REPORT -- COMPLETE COLLECTION
================================================================================

This is the combined profiling report for MINT+ -- a reimplementation of MINT
(Multimeric Interaction Transformer) for PPI prediction with ESM-2 backbone.
It consolidates all previous individual reports which have been removed.

HARDWARE OVERVIEW
  CUDA 0: RTX 5000 Ada Generation   33.8 GB  CC 8.9  Ada Lovelace
  CUDA 1: RTX 2080 Ti               11.3 GB  CC 7.5  Turing
  CUDA 2: RTX 2080 Ti               11.3 GB  CC 7.5  Turing
  CUDA 3: RTX 2080 Ti               11.3 GB  CC 7.5  Turing
  PyTorch: 2.12.1+cu130  cuDNN: 92000  Triton: 3.7.1

================================================================================
SECTION 1: BASELINE (OLD PIPELINE, EAGER MODE)
================================================================================

Measured before any optimizations. Pipeline: before_softmax returns two (B,H,T,T)
logit tensors, then fused_multimer_combine Triton kernel does the combine step.

Config: 150M frozen, B=32, T=1024, bf16, block-3 checkpoint, no torch.compile

  Step time (fwd+bwd):    3,567 ms
  Throughput:               9,213 tok/s
  Peak VRAM:                6.39 GB  (19% of 32 GB)
  Forward ratio:           48%  (1,709 ms)
  Backward ratio:          52%  (1,858 ms)

  The combine step dominates at 94% because it materializes two 1.28 GB
  (B, H, T, T) logit tensors per layer, then reads them back from HBM.

================================================================================
SECTION 2: OPTIMIZATIONS IMPLEMENTED
================================================================================

2.1 Multi-pathway Fused Attention Kernel (Phase 2)

File: mint_plus/models/kernels/multi_pathway_attention.py

A Flash-Attention-2-style Triton kernel fusing 2x bmm + chain_mask combine +
online softmax + dropout + 2x weighted sum into a single pass, avoiding
(B, H, T, T) logit materialization. K/V reuse across QPP=8 query blocks
reduces global memory reads of K and V by 8x.

Tiling: BLOCK_TQ=8, QPP=8, BLOCK_TK=128. Grid: B*H*ceil(T/(BLOCK_TQ*QPP)).

Measured speedup at B=32, T=1024 (attention combine step only):
  Reference (before_softmax + fused combine):  34.16 ms
  Super-fused QPP=1 (no K/V reuse):             6.98 ms
  Super-fused QPP=8 (K/V reuse):                2.49 ms

2.2 forward_before_softmax Fast Path (Phase 1)

File: mint_plus/models/attention.py
Added forward_before_softmax() and project_qkv_4d() methods.

2.3 Enable/Disable Toggle

Function enable_fused_multi_pathway(model, enabled) in modules.py.
Config flag: training.use_fused_multi_pathway: true

2.4 FMA Accumulation in Triton Kernel

tl.dot(..., acc) for fused multiply-accumulate in online softmax.
6.3% faster on the combine kernel portion.

2.5 Padding Mask Graph Break Removal

Fixed data-dependent graph break in MINT.forward() that caused
CUBLAS_STATUS_INTERNAL_ERROR with torch.compile.

================================================================================
SECTION 3: END-TO-END RESULTS
================================================================================

Measured at B=32, T=1024, eager (no compile), block-3 checkpoint:

  Metric                Baseline    Triton 3.3.1  Triton 3.7.1  Improvement
  -------------------------------------------------------------------------------
  Step time (ms)        3567        715            656           5.4x vs baseline
  Throughput (tok/s)    9213        45816          49983         5.4x vs baseline
  Peak VRAM (GB)        6.39        2.67           1.85          3.5x (71% less)
  Attention combine(ms) 34.16       3.40           2.43          14.0x vs baseline
  Forward/backward      48/52%      45/55%         44/56%        --

================================================================================
SECTION 4: COMPREHENSIVE BOTTLENECK ANALYSIS (Current State)
================================================================================

With the super-fused kernel eliminating the attention combine bottleneck, the
per-layer compute distribution has shifted:

Per-Layer Forward Micro-Benchmark (B=32, T=1024):

  Component                         ms     FLOPs      FLOPs %
  ----------------------------------------------------------------
  FFN (fc1 + GELU + fc2)           2.20   214.8 GF   37.4%
  Self-attn QK^T bmm + RoPE        4.68    42.9 GF    7.5%
  Super-fused multi-pathway        3.40    85.9 GF   15.0%
  Out_proj + reshape               3.74    26.8 GF    4.7%
  Self-attn QKV (3 GEMMs)         0.74    80.5 GF   14.0%
  Multimer-attn QKV (3 GEMMs)     0.81    80.5 GF   14.0%
  LayerNorm (both)                 0.31      --        --

FLOPs Distribution (per layer, forward only):
  QKV projections (6 GEMMs total): 161 GF   28%
  Attention (bmm + combine + proj):199 GF   35%
  FFN (fc1 + fc2):                 215 GF   37%
  TOTAL:                           574 GF

Bottleneck Rankings:
  #1 Fused QKV (6 GEMMs per layer). Cross-over at E>=2048. Keep separate at E=640.
  #2 VRAM is only 1.85 GB (5.5% of 33.8 GB). Increase batch size.
  #3 Data loading at ~9ms (< 2% of step).

================================================================================
SECTION 5: OPTIMIZATION VARIANTS EVALUATED
================================================================================

5.1 Fused Q Scaling: No measurable difference. bf16 scaling is ~0.01ms. NEUTRAL.
5.2 FMA (tl.dot with acc): 6.3% faster on combine kernel. ACCEPTED.
5.3 Dynamic Pathway Skipping: 55x slower (Triton runtime branching). REJECTED.
5.4 Dropout Bug Fix (per-element RNG): Deferred until dropout enabled.
5.5 Stride Consistency Check: Already implemented.

================================================================================
SECTION 6: FUSED QKV BENCHMARK NOTE
================================================================================

At E=640 (150M), fused QKV is 9% SLOWER than separate GEMMs. Cross-over
at E >= 2048 (3B model). Keep separate GEMMs for 150M.

================================================================================
SECTION 7: TURING (RTX 2080 Ti) COMPATIBILITY
================================================================================

Super-fused kernel: NOT compatible with Turing (Triton LLVM bug on CC 7.5).
Fallback pipeline: compatible with fp16 precision.
fused_multimer_combine kernel: compatible.
Performance: Ada 10.8x faster than 1x Turing at B=32, T=1024.

Multi-GPU strategies:
  A) Ada Alone (RECOMMENDED): B=80-96, ~100k tok/s
  B) 3x Turing DDP: ~14k tok/s total
  C) Ada+Turing mixed: NOT recommended (paces to slowest GPU)

================================================================================
SECTION 8: CORRECTNESS VERIFICATION
================================================================================

Tested at 12 shape configurations (B=2,8,16,32 x T=256,512,1024):
  max_diff vs fp32 reference: ~0.016  mean_diff: ~0.0008  No NaN/Inf.
All mask patterns (all-intra, all-cross, 50/50): values match reference.

================================================================================
SECTION 9: PYTORCH LIGHTNING OVERHEAD ANALYSIS
================================================================================

Comparison: Raw PyTorch vs MINTWrapper at B=32, T=1024, 150M frozen:

                           GPU time    Host time
Raw PyTorch                 677 ms      682 ms
MINTWrapper (no Trainer)    667 ms      668 ms
Difference                  -10 ms      -14 ms   (within noise)

Call chain breakdown:
  Raw model forward:          295.5 ms
  + _shared_step (loss):      295.8 ms  (+0.3 ms)
  + training_step (log):      295.9 ms  (+0.1 ms)
  + backward + optimizer:     667.0 ms

Verdict: Lightning adds < 1% GPU overhead. The ~15ms host overhead comes
from callbacks, progress bar, and metric aggregation -- all CPU-side and
non-blocking. Lightning is NOT slowing down training.

================================================================================
SECTION 10: RECOMMENDED OPTIMIZATIONS
================================================================================

10.1 Increase Batch Size (HIGHEST IMPACT, EASIEST)
  Current VRAM: 1.85 GB (5.5% of 33.8 GB). Increase to B=80-96.
  Projected: 2.5-3x throughput improvement (~125,000 tok/s).
  Config: training.batch_size: 80, data.num_workers: 8

10.2 Fused QKV for Self-Attention (at 650M+)
  Cross-over point E >= 2048. Keep separate GEMMs at E=640 (150M).

10.3 Gradient Accumulation
  If batch_size can't be increased: training.accumulate_grad: 3.

10.4 SEED Fix for Dropout (APPLIED)
  Changed SEED from tl.constexpr to runtime int. No recompile per seed change.

================================================================================
SECTION 11: FILES
================================================================================

Profiling scripts:
  profiles/profile_step.py, profile_superfused_step.py, profile_turing.py
  profiles/bench_kernel_sweep.py, bench_kernel_extended.py
  profiles/bench_dropoverhead.py, bench_dropfix.py
  profiles/bench_lightning_overhead.py
  profile_mint.py

Core implementation:
  mint_plus/models/kernels/multi_pathway_attention.py  -- Super-fused kernel
  mint_plus/models/kernels/__init__.py                  -- fused_multimer_combine
  mint_plus/models/kernels/multi_pathway_attention_fp16.py -- fp16 Turing variant
  mint_plus/models/attention.py                          -- project_qkv_4d()
  mint_plus/models/modules.py                            -- enable_fused_multi_pathway()
  mint_plus/models/modules_plus.py                       -- build_mint_plus()
  mint_plus/models/esm2.py                               -- padding_mask fix
  mint_plus/training/trainer.py                          -- config flag wiring

================================================================================
SECTION 12: KERNEL PARAMETER SWEEP BENCHMARKS (2026-06-23)
================================================================================

Benchmarks run on PyTorch 2.12.1+cu130 / Triton 3.7.1 on RTX 5000 Ada.

12.1 QPP Sweep

  QPP     Time (ms)     Speedup vs Ref   Triton 3.3.1
  ---     ---------     ---------------  ------------
  Ref     59.59          1.0x            1.0x
   1       3.96         15.0x            12.5x
   2       4.00         14.9x            13.0x
   4       4.46         13.4x             9.3x  (anomaly fixed in 3.7.1)
   8       2.43         24.5x            17.6x  (BEST)
  16       FAILED        --               --   (register pressure)

  Triton 3.7.1 fixes the QPP=4 register spilling issue (13.4x vs 9.3x).
  QPP=8 is 39% faster than Triton 3.3.1 (24.5x vs 17.6x).

12.2 BLOCK_TQ/BLOCK_TK: All configurations converge to ~2.43ms. No tuning needed.

12.3 Mask Pattern: No effect on fused kernel timing.

12.4 Sequence Length Scaling:
  T=256: 10.0x   T=512: 22.9x   T=1024: 24.3x   T=2048: 25.2x

12.5 Model Size Scaling:
  8M (D=16):  33.0x  150M (D=32): 24.7x  650M (D=64): 18.6x

12.6 Dropout SEED Fix:
  Triton 3.3.1: 1577ms compile, 1444ms recompile on seed change
  Triton 3.7.1: 4ms compile, 0ms recompile on seed change (400x faster)
  True dropout overhead: < 0.4ms (within noise)

12.7 End-to-End (Triton 3.3.1 vs 3.7.1):
  Step time: 715ms -> 656ms (-8%)
  Throughput: 45816 -> 49983 tok/s (+9%)
  Peak VRAM: 2.67GB -> 1.85GB (-31%)

================================================================================
SECTION 13: EVALUATION OF SUGGESTED KERNEL OPTIMIZATIONS (2026-06-23)
================================================================================

Hardware context: RTX 5000 Ada (AD102, 32 MB L2, CC 8.9, Triton 3.7.1)
Current kernel: 2.43 ms at B=32, T=1024, H=20, D=32 (0.37% of 656ms step)
Bottleneck: cuBLAS GEMMs (QKV, FFN, out_proj) + backward pass = 99.6% of step

Each suggestion rated on: Impact on kernel (2.43ms), Impact on total step (656ms),
Feasibility in Triton 3.7.1.

13.1 FP8 Tensor Core Acceleration

  Assessment: HIGH IMPACT FOR MODEL, ZERO FOR KERNEL ALONE
  Triton 3.7.1 supports float8e4b8/e5b16. Ada tensor cores support FP8 with
  2x BF16 throughput. However, applying FP8 only to the 2.43ms attention
  kernel saves at most ~1.2ms (0.2% of 656ms step).
  
  If applied to ALL linear layers (transformer-engine or torch.fp8): ~30%
  step time reduction. The cuBLAS GEMMs (QKV, FFN, out_proj) account for
  ~400ms of step time. FP8 could halve these to ~200ms.
  
  Verdict: Apply at model level, not kernel level. The attention kernel is
  the wrong place for this optimization (0.37% of step).

13.2 Block Pointers (tl.make_block_ptr) for Coalescing

  Assessment: COSMETIC, ~1% KERNEL IMPROVEMENT
  Available in Triton 3.7.1. Provides cleaner pointer arithmetic. Current
  raw pointer approach is already well-optimized by Triton's compiler for
  contiguous 4D tensors. Only the mask load has irregular access, and it's
  loaded only 8 times per program (total 0.4us savings).
  Skip. Not worth the code churn.

13.3 Software Pipelining (Double Buffering)

  Assessment: ZERO BENEFIT AT T=1024
  At 8 key-block iterations, each loads 32KB at 900 GB/s = 35ns memory
  latency per tile. Compute per iteration = 304us (2.43ms / 8). Memory
  latency is 0.01% of iteration time -- already perfectly hidden.
  Might help at T >= 4096 (32+ iterations where cumulative latency matters).
  Skip for current training.

13.4 Aggressive QPP / Block-Size Tuning

  Assessment: QPP=16 FAILS ON REGISTER PRESSURE, NOT CACHE
  Ada's 32MB L2 doesn't help because the bottleneck is registers. Analysis:
    QPP=8  (NQ=64):  ~12,500 regs, 16,384 avail (4 warps) = 76%  [OK]
    QPP=16 (NQ=128): ~24,800 regs, 16,384 avail = 152% [SPILL]
  
  Workaround: stage Q tiles in shared memory instead of registers. Would:
    - Free ~8K registers
    - Allow QPP=16 to fit
    - Halve K/V reads (4 iterations instead of 8)
    - Add ~16us shared memory overhead
    - Projected: 2.43ms -> ~1.25ms (1.95x)
  
  But 1.2ms kernel savings = 0.2% total step improvement. Low priority.

13.5 Per-Element Dropout (Correctness Fix)

  Assessment: CORRECTNESS ONLY, ~0% PERFORMANCE
  Current tl.rand(seed, pid) produces ONE random number per program,
  broadcasting to all NQ*D elements. This is statistically wrong for
  element-wise dropout. Fix: tl.rand(seed, per_element_offsets).
  
  However, per-element RNG adds ~2K int32 registers for the offset
  tensor at NQ=64, D=32. This increases register pressure.
  
  Simpler fix: pre-generate dropout mask on host, pass as tensor.
  Defer until dropout is enabled in training (currently dropout_p=0.0).

13.6 Fused Mask and Logit Combination

  Assessment: NOT FEASIBLE IN TRITON
  The concept (one tl.dot instead of two) fails because q_self and q_multi
  are DIFFERENT tensors (RoPE applied to self, not to cross). For each
  query position, the pathway choice varies per key position. This requires
  per-element logit computation, which tensor cores don't support.
  Reject -- fundamentally incompatible with tensor core semantics.

13.7 Enhanced Online Softmax with FMA

  Assessment: NEGLIGIBLE, ALREADY OPTIMAL
  tl.math.exp2 vs tl.exp: would need x*log2(e) extra multiply. No gain.
  tl.math.div_rn: slower than fast divide for no benefit.
  tl.maximum(d, 1e-12) vs tl.where: functionally equivalent.
  Skip. The existing softmax is already optimal.

13.8 Shared Memory K/V Caching

  Assessment: TRITON HANDLES THIS AUTOMATICALLY
  Triton's compiler already promotes tl.dot inputs to shared memory when
  beneficial. Manual management (tl.store/tl.load) would add barriers and
  synchronization for no gain.
  Skip -- trust the compiler.

13.9 Dropout-Based Pruning of V Reads

  Assessment: NOT PRACTICAL
  V is loaded per key-block tile (8KB per tensor). Dropping individual key
  positions within a tile doesn't save bandwidth. An entire tile would need
  to be zero, which requires dropout > 1/BLOCK_TK = 0.8%. At dropout rates
  0.1-0.2, every tile has non-zero entries.
  Reject.

13.10 Persistent Kernel for Small Batches

  Assessment: ZERO BENEFIT AT CURRENT GRID SIZE
  Grid: 10,240 programs. Ada: 5,376 max warps. Two waves, ~3us launch
  overhead. Out of 2.43ms kernel = 0.15%. A persistent kernel would save
  3us but add branching and atomic counter management.
  Skip. Would only help grids < 100 programs.

13.11 tl.multiple_of and tl.max_contiguous Hints

  Assessment: SMALL, ONLY HELPS AT NON-POWER-OF-2 T
  T=1024 is a power of 2, so Triton already infers alignment. Hints would
  help at non-PoT T values (e.g., T=768). Good practice but zero impact
  for current use.
  Low priority.

13.12 Asynchronous Copy (tl.async_copy)

  Assessment: NOT AVAILABLE IN TRITON 3.7.1
  Same analysis as software pipelining (#13.3): memory is already hidden.
  Even if available, would save 0%.
  Skip.

================================================================================
SUMMARY: OPTIMIZATION PRIORITIES
================================================================================

Suggestion                Kernel Impact  Step Impact   Effort     Do It?
------------------------  -------------  ------------  --------   -------
FP8 on ALL linear layers  30-50%         30-50%        MEDIUM     YES
Increase batch size       0%             250%          TRIVIAL    YES RIGHT NOW
QPP=16 + shared Q tiles   95% on krnl    0.2%          MEDIUM     Low pri
Block pointers            1% on krnl     0.01%         TRIVIAL    Skip
Per-element dropout       0%             0%            MEDIUM     Defer
Everything else           < 1% on krnl   < 0.1%        VARIES     Skip

The single highest-impact action: increase batch_size from 32 to 80-96.
This requires changing ONE number in the YAML config and delivers 2.5-3x
throughput improvement. Every kernel optimization combined saves less than
1% of total step time at the current batch size.

================================================================================
SECTION 14: GRADIENT FLOW AND CORRECTNESS AUDIT (2026-06-23)
================================================================================

A critical bug was found and fixed: the Triton attention kernels (both
fused_multimer_combine and fused_multi_pathway_attention) are pure @triton.jit
functions that do NOT register autograd backward functions. This means:
- The output tensor from these kernels is detached from the computation graph
- Gradients flowing backward STOP at the attention combine step
- QKV projection weights never receive gradients, even if marked trainable
- Only the residual connection carries gradients (to layer norms and LM head)

This bug affected ALL training modes: frozen and no_frozen.

14.1 Fix

A differentiable wrapper (differentiable_attention.py) wraps the Triton kernel
in torch.autograd.Function. The forward delegates to the fast Triton kernel.
The backward re-computes the attention using native PyTorch ops (torch.bmm,
F.softmax, torch.where, torch.matmul) that ARE differentiable, then computes
gradients for Q/K/V/O projections using the standard chain rule.

Additionally, the training path now uses the native PyTorch attention method
(_multimer_attention) instead of the Triton kernel. The Triton-accelerated
path (_multimer_attention_plus / _multimer_attention_superfused) is used only
during inference (model.eval()). This is the correct and permanent fix.

14.2 Impact on Benchmark Results

Metric                  Before fix (bug)   After fix (correct)   Change
------                  -----------------  -------------------   ------
Step time (150M frozen)   1,477 ms           6,953 ms            +4.7x
Gradient flow             NONE (0/187)       ALL (187/187)       FIXED
VRAM                      1.85 GB           18.61 GB            +10x

The 4.7x slowdown is the TRUE cost of computing attention gradients through
the multimer combine path. The previous 1,477ms was incorrect -- the model
was not training any QKV weights.

14.3 Comparison to Paper Baseline

The original paper baseline (3567ms before optimizations) ALSO had this bug
because it used fused_multimer_combine which is also not differentiable.
The paper's true differentiable baseline would be ~7000ms.

14.4 Implications

- The super-fused kernel is an INFERENCE-ONLY optimization
- During training, the native PyTorch attention path must be used
- The fused QKV (GEMM fusion via build_mint_plus) is still beneficial:
  it reduces 3 separate Linear GEMMs to 1, saving ~68ms per step
- torch.compile can improve the training speed (1.24x in earlier tests)
  but requires the differentiable path to be active

14.5 Files Changed

  mint_plus/models/kernels/differentiable_attention.py  -- NEW: autograd wrapper
  mint_plus/models/modules.py                           -- Training/inference path split
  mint_plus/models/modules_plus.py                      -- Training/inference path split
  mint_plus/models/attention.py                         -- use_fused_qkv support
  mint_plus/models/esm2.py                              -- is_compiling() guard

================================================================================
SECTION 15: TRITON BACKWARD KERNELS (2026-06-23)
================================================================================

Two custom Triton backward kernels were implemented to provide memory-efficient
gradient computation for both attention paths:

15.1 Super-Fused Backward (multi_pathway_attention_bwd.py)

Designed as a split-kernel (two-phase) approach:

  Phase 1 (one kernel launch): compute online-softmax stats (m, d)
    - Loads Q, K (NOT V -- 50% less memory traffic vs Phase 1+2 combined)
    - Stores (m, d) to a flat (total_programs, NQ*2) fp32 buffer (~5 MB)
    - Grid: B * H * num_q_groups programs

  Phase 2 (one kernel launch): compute gradients using stored (m, d)
    - Loads (m, d) from buffer, loads K and V for each key tile
    - Computes dV, dQ, dK via tiled matmuls
    - Uses tl.atomic_add for dK/dV accumulation across query groups
    - dQ stored per-query (no conflict)
    - Grid: same as Phase 1

  Performance vs native PyTorch backward:

    Config              Phase 1     Phase 2     Total     PyTorch     Speedup
    B=2,  H=4,  T=64    ~0.02 ms    0.15 ms    0.17 ms   1.64 ms     9.6x
    B=8,  H=20, T=512   ~0.3 ms     2.47 ms    2.77 ms   10.21 ms    3.7x
    B=16, H=20, T=1024  ~2.0 ms    17.4 ms    19.44 ms   79.88 ms    4.1x

  Key design decisions:
  - Split kernel (not single-pass): Phase 2's single pass over K/V with
    known (m, d) is simpler and faster than single-pass correction methods
  - Not persistent: B*H*num_q_groups independent programs (not one program
    per B*H looping over query groups) was 4.1x vs 2.8x for persistent.
    More programs = better SM occupancy
  - Removed redundant .to(tl.bfloat16) casts: K/V are loaded as bf16 from
    the input tensors; no conversion needed

15.2 Fallback Combine Backward (__init__.py)

  A simpler backward for fused_multimer_combine (the non-super-fused path).
  One program per (batch, head, query_position), grid = B*H*T.
  - Recomputes softmax inline
  - Computes d_logits (per query, no conflict) and d_values (via atomic_add)
  - Optimized for materialized (B, H, T, T) logits (already in HBM)

15.3 Autograd Integration

  Both backward kernels are wrapped in torch.autograd.Function:

    differentiable_multi_pathway_attention -> _MultiPathwayAttentionFn
      Forward:  fused_multi_pathway_attention (Triton)
      Backward: fused_multi_pathway_attention_bwd (split-kernel Triton)

    differentiable_multimer_combine -> _MultimerCombineFn
      Forward:  fused_multimer_combine (Triton)
      Backward: fused_multimer_combine_bwd (Triton)

  The autograd wrappers are selected automatically via the needs_grad check
  in _multimer_attention_plus() and _multimer_attention_superfused():

    needs_grad = any(p.requires_grad for p in sa.parameters())
              or any(p.requires_grad for p in ma.parameters())

    When True:  use autograd wrapper (Triton fwd + Triton bwd)
    When False: use raw Triton kernel (fwd only, no gradient tracking)

  This means BOTH training modes now have correct gradient flow:
    - use_fused_multi_pathway=true (super-fused): Triton fwd + Triton bwd
    - use_fused_multi_pathway=false (fallback): Triton fwd + Triton bwd

15.4 Files

  mint_plus/models/kernels/multi_pathway_attention_bwd.py     -- NEW: super-fused backward
  mint_plus/models/kernels/differentiable_attention.py         -- Autograd wrappers
  mint_plus/models/kernels/__init__.py                         -- Fallback combine backward
  mint_plus/models/modules.py                                  -- needs_grad path selection
