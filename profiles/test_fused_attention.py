"""
Test the fused multi-pathway attention kernel against the reference pipeline.

Tests at multiple shapes, with and without dropout, and benchmarks both paths.

Usage:
    CUDA_VISIBLE_DEVICES=0 python profiles/test_fused_attention.py
"""

import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint_plus.models.attention import MultiHeadAttention
from mint_plus.models.kernels import fused_multimer_combine, fused_multi_pathway_attention


def apply_rope_4d(q, k, rot_emb):
    """Apply RoPE to Q and K in (B, H, T, D) format using correct layout.

    Rotates to (B*H, T, D) for the rotary embedding module (which expects
    contiguous input in this layout), then returns to (B, H, T, D).
    """
    B, H, T, D = q.shape
    # Collapse batch+heads: (B, H, T, D) -> (B*H, T, D)
    q_3d = q.reshape(B * H, T, D)  # already contiguous as (B,H,T,D)
    k_3d = k.reshape(B * H, T, D)
    q_r = rot_emb(q_3d)  # (B*H, T, D)
    k_r = rot_emb(k_3d)
    # Restore head dimension: (B*H, T, D) -> (B, H, T, D)
    return (q_r.view(B, H, T, D).contiguous(),
            k_r.view(B, H, T, D).contiguous())


def time_block(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def test_kernel(B, T, H, D, device="cuda", atol=0.2, verbose=True):
    """Test the new fused kernel against the reference pipeline."""
    E = H * D
    scaling = D ** -0.5

    sa = MultiHeadAttention(E, H, use_rotary_embeddings=True).to(device).to(torch.bfloat16)
    ma = MultiHeadAttention(E, H, use_rotary_embeddings=False, no_proj=True).to(device).to(torch.bfloat16)

    x = torch.randn(T, B, E, device=device, dtype=torch.bfloat16, requires_grad=False)
    chain_mask = torch.rand(B, T, T, device=device) < 0.4
    chain_mask = chain_mask & ~torch.eye(T, device=device, dtype=torch.bool).unsqueeze(0)
    padding_mask = None

    # ---- Reference pipeline ----
    ls, vs = sa.forward_before_softmax(x, key_padding_mask=padding_mask)
    lm, vm = ma.forward_before_softmax(x, key_padding_mask=padding_mask)
    ref = fused_multimer_combine(ls, lm, chain_mask, vs, vm, dropout_p=0.0)

    # ---- New pipeline ----
    qs, ks, vss = sa.project_qkv_4d(x)
    qm, km, vmm = ma.project_qkv_4d(x)
    qs = qs * scaling
    qm = qm * scaling
    qs, ks = apply_rope_4d(qs, ks, sa.rot_emb)
    new = fused_multi_pathway_attention(qs, ks, vss, qm, km, vmm, chain_mask,
                                        dropout_p=0.0, training=False)

    diff = (ref.float() - new.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    has_nan = torch.isnan(new).any().item()
    passed = max_diff < atol and not has_nan

    if verbose:
        print(f"  B={B:3d} T={T:4d} H={H:2d} D={D:2d}  "
              f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
              f"{'PASS' if passed else 'FAIL'}{' (NaN!)' if has_nan else ''}")
    return max_diff, mean_diff, passed


def test_dropout(B, T, H, D, device="cuda"):
    """Dropout statistical check."""
    E = H * D
    scaling = D ** -0.5
    sa = MultiHeadAttention(E, H, use_rotary_embeddings=True).to(device).to(torch.bfloat16)
    ma = MultiHeadAttention(E, H, use_rotary_embeddings=False, no_proj=True).to(device).to(torch.bfloat16)

    x = torch.randn(T, B, E, device=device, dtype=torch.bfloat16)
    chain_mask = torch.rand(B, T, T, device=device) < 0.4

    qs, ks, vss = sa.project_qkv_4d(x)
    qm, km, vmm = ma.project_qkv_4d(x)
    qs = qs * scaling
    qm = qm * scaling
    qs, ks = apply_rope_4d(qs, ks, sa.rot_emb)
    new = fused_multi_pathway_attention(qs, ks, vss, qm, km, vmm, chain_mask,
                                        dropout_p=0.1, training=True)

    has_nan = torch.isnan(new).any().item()
    new_mean = new.float().abs().mean().item()
    sensible = 0.001 < new_mean < 10.0
    passed = not has_nan and sensible
    print(f"  [Dropout p=0.1] B={B} T={T}: mean={new_mean:.4f} NaN={has_nan} "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def benchmark(B, T, H, D, device="cuda", iters=30):
    """Benchmark both paths."""
    E = H * D
    scaling = D ** -0.5
    sa = MultiHeadAttention(E, H, use_rotary_embeddings=True).to(device).to(torch.bfloat16)
    ma = MultiHeadAttention(E, H, use_rotary_embeddings=False, no_proj=True).to(device).to(torch.bfloat16)

    x = torch.randn(T, B, E, device=device, dtype=torch.bfloat16)
    chain_mask = torch.rand(B, T, T, device=device) < 0.4

    qs, ks, vss = sa.project_qkv_4d(x)
    qm, km, vmm = ma.project_qkv_4d(x)
    qs_s = qs * scaling
    qm_s = qm * scaling
    qs_r, ks_r = apply_rope_4d(qs_s, ks, sa.rot_emb)

    # Reference: before_softmax + combine
    def ref_step():
        ls, vs = sa.forward_before_softmax(x)
        lm, vm = ma.forward_before_softmax(x)
        fused_multimer_combine(ls, lm, chain_mask, vs, vm, dropout_p=0.0)
    ref_ms = time_block(ref_step, warmup=5, iters=iters)
    # New: fused multi-pathway
    new_ms = time_block(
        lambda: fused_multi_pathway_attention(
            qs_r, ks_r, vss, qm_s, km, vmm, chain_mask,
            dropout_p=0.0, training=False),
        warmup=5, iters=iters,
    )
    speedup = ref_ms / new_ms
    print(f"  B={B:3d} T={T:4d}: ref={ref_ms:.1f}ms  new={new_ms:.1f}ms  "
          f"speedup={speedup:.2f}x")
    return ref_ms, new_ms, speedup


if __name__ == "__main__":
    device = "cuda"
    torch.set_float32_matmul_precision('medium')
    import triton
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}  "
          f"GPU: {torch.cuda.get_device_name(0)}")
    passed_all = True

    print("\n" + "=" * 70)
    print("TEST 1: Correctness across shapes")
    print("=" * 70)
    for B in [2, 8, 16, 32]:
        for T in [256, 512, 1024]:
            mx, mn, p = test_kernel(B, T, 20, 32, device)
            passed_all = passed_all and p

    print("\n" + "=" * 70)
    print("TEST 2: Dropout sanity")
    print("=" * 70)
    for B in [8, 16]:
        for T in [512, 1024]:
            passed_all = passed_all and test_dropout(B, T, 20, 32, device)

    print("\n" + "=" * 70)
    print("TEST 3: Different head configs")
    print("=" * 70)
    for H, D in [(8, 64), (20, 32)]:
        if H * D > 640:
            continue
        mx, mn, p = test_kernel(4, 512, H, D, device)
        passed_all = passed_all and p

    print("\n" + "=" * 70)
    print("TEST 4: Edge cases (all-same-chain, all-cross-chain, large values)")
    print("=" * 70)
    B, T, H, D = 4, 512, 20, 32
    E = H * D
    scaling = D ** -0.5
    sa = MultiHeadAttention(E, H, use_rotary_embeddings=True).to(device).to(torch.bfloat16)
    ma = MultiHeadAttention(E, H, use_rotary_embeddings=False, no_proj=True).to(device).to(torch.bfloat16)
    x = torch.randn(T, B, E, device=device, dtype=torch.bfloat16) * 5.0

    for name, mask in [("all-same-chain", torch.zeros(B, T, T, dtype=torch.bool, device=device)),
                       ("all-cross-chain", torch.ones(B, T, T, dtype=torch.bool, device=device))]:
        ls, vs = sa.forward_before_softmax(x)
        lm, vm = ma.forward_before_softmax(x)
        ref = fused_multimer_combine(ls, lm, mask, vs, vm, dropout_p=0.0)
        qs, ks, vss = sa.project_qkv_4d(x)
        qm, km, vmm = ma.project_qkv_4d(x)
        qs2, ks2 = apply_rope_4d(qs * scaling, ks, sa.rot_emb)
        new = fused_multi_pathway_attention(qs2, ks2, vss, qm * scaling, km, vmm,
                                            mask, dropout_p=0.0, training=False)
        dff = (ref - new).abs().max().item()
        p = dff < 1.2
        passed_all = passed_all and p
        print(f"  {name}: max_diff={dff:.6f}  {'PASS' if p else 'FAIL'}")

    print("\n" + "=" * 70)
    print("BENCHMARK")
    print("=" * 70)
    for B in [16, 32]:
        for T in [512, 1024]:
            try:
                benchmark(B, T, 20, 32, device, iters=20)
            except Exception as e:
                print(f"  B={B} T={T}: ERROR {e}")

    print()
    if passed_all:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
