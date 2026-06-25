"""Profile the full training step with the super-fused kernel enabled."""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint_plus.models.esm2 import MINT
from mint_plus.models.modules import build_checkpointed_model, enable_fused_multi_pathway
from mint_plus.models import MODEL_REGISTRY

torch.set_float32_matmul_precision('medium')

def time_block(fn, warmup=5, iters=20, name="block"):
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
    ms = start.elapsed_time(end) / iters
    return ms

def build_model():
    config = MODEL_REGISTRY["150M"]
    model = MINT(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        use_multimer=True,
    )
    ckpt = "./ckpts/esm2_150M/pytorch_model.bin"
    if os.path.exists(ckpt):
        print(f"Loading pretrained weights from {ckpt}")
        model.load_pretrained_weights(ckpt, dtype=torch.bfloat16)
    
    # Apply freeze_self_attn like the training config
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        if 'multimer_attn' in name or 'lm' in name or 'norm' in name:
            p.requires_grad = True
    
    # Enable super-fused kernel
    enable_fused_multi_pathway(model, enabled=True)
    print("Super-fused multi-pathway kernel ENABLED")
    
    # Block-3 checkpointing
    model = build_checkpointed_model(model, block_size=3)
    model = model.cuda().to(torch.bfloat16)
    model.train()
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
    return model

def profile():
    device = "cuda"
    B, T = 32, 1024
    L = 30
    
    print(f"Model: 150M frozen, B={B}, T={T}")
    print(f"Super-fused kernel: ON")
    print(f"Block checkpointing: ON (block_size=3)")
    print()
    
    model = build_model()
    
    # Synthetic data
    tokens = torch.randint(4, 24, (B, T), device=device)
    tokens[:, 0] = 0      # CLS
    tokens[:, -1] = 2     # EOS
    chain_ids = torch.zeros(B, T, dtype=torch.int32, device=device)
    chain_ids[:, 512:] = 1  # Two chains of 512 each
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        out = model(tokens, chain_ids=chain_ids)['logits']
        loss = F.cross_entropy(out.transpose(1, 2), tokens, reduction='none')
        loss = loss.sum()
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    
    # Fwd + bwd timing
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad()
    torch.cuda.synchronize()
    
    def step():
        out = model(tokens, chain_ids=chain_ids)['logits']
        loss = F.cross_entropy(out.transpose(1, 2), tokens, reduction='none')
        loss = loss.sum()
        loss.backward()
    
    ms_fwd_bwd = time_block(step, warmup=3, iters=10, name="fwd_bwd")
    peak = torch.cuda.max_memory_allocated() / 1e9
    tok_s = B * T * 1000 / ms_fwd_bwd
    print(f"\n  Forward+Backward: {ms_fwd_bwd:.1f} ms")
    print(f"  Throughput:       {tok_s:.0f} tok/s")
    print(f"  Peak VRAM:        {peak:.2f} GB")
    
    # Forward only
    def fwd_only():
        out = model(tokens, chain_ids=chain_ids)
        out["logits"].sum()
    
    ms_fwd = time_block(fwd_only, warmup=3, iters=20, name="fwd")
    print(f"  Forward only:     {ms_fwd:.1f} ms ({ms_fwd/ms_fwd_bwd*100:.0f}%)")
    print(f"  Backward only:    {ms_fwd_bwd-ms_fwd:.1f} ms ({(ms_fwd_bwd-ms_fwd)/ms_fwd_bwd*100:.0f}%)")
    
    # Memory breakdown
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Memory breakdown:")
    print(f"    Model params (bf16):         {total*2/1e9:.2f} GB")
    print(f"    Gradients (trainable):       {trainable*2/1e9:.2f} GB")
    print(f"    Adam states (fp32):          {trainable*4*2/1e9:.2f} GB")
    print(f"    Base footprint:              {(total*2 + trainable*2 + trainable*8)/1e9:.2f} GB")
    print(f"    Peak measured:               {peak:.2f} GB")
    
    # Per-layer micro-benchmark
    print(f"\n{'='*60}")
    print("PER-LAYER MICRO-BENCHMARK (B=32, T=1024)")
    print(f"{'='*60}")
    
    block0 = model.layers[0]
    layer = block0.layers[0]
    print(f"  Layer: {type(layer).__name__}")
    
    x = torch.randn(T, B, 640, device=device, dtype=torch.bfloat16, requires_grad=True)
    padding_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
    
    # Build chain mask for micro-benchmark
    c = chain_ids[:B]
    chain_mask = ~torch.eq(c.unsqueeze(-1), c.unsqueeze(-2))
    
    sa = layer.self_attn
    ma = layer.multimer_attn
    ff = layer.feed_forward
    ln1 = layer.self_attn_layer_norm
    ln2 = layer.final_layer_norm
    
    comps = {}
    
    def time_comp(fn, name, warmup=10, iters=200):
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
        comps[name] = start.elapsed_time(end) / iters
    
    # LayerNorm
    time_comp(lambda: (ln1(x), ln2(x)), "layernorm (both)", iters=500)
    
    h_ln = ln1(x)
    
    # Self-attn QKV projections (3 separate GEMMs)
    time_comp(lambda: (sa.q_proj(h_ln), sa.k_proj(h_ln), sa.v_proj(h_ln)),
              "self_attn QKV (3 GEMMs)", iters=300)
    
    # Multimer-attn QKV (3 separate GEMMs, no fused QKV in current code)
    time_comp(lambda: (ma.q_proj(h_ln), ma.k_proj(h_ln), ma.v_proj(h_ln)),
              "multimer_attn QKV (3 GEMMs)", iters=300)
    
    # Self-attn BMM + RoPE
    def self_bmm():
        q = sa.q_proj(h_ln) * sa.scaling
        k = sa.k_proj(h_ln)
        q = q.view(T, B*20, 32).transpose(0, 1)
        k = k.view(T, B*20, 32).transpose(0, 1)
        q, k = sa.rot_emb(q), sa.rot_emb(k)
        return torch.bmm(q, k.transpose(1, 2))
    time_comp(self_bmm, "self_attn bmm + RoPE", iters=100)
    
    # Super-fused multi-pathway attention
    from mint_plus.models.kernels.multi_pathway_attention import fused_multi_pathway_attention
    
    # Prepare inputs for super-fused kernel
    qs, ks, vss = sa.project_qkv_4d(h_ln)
    qm, km, vmm = ma.project_qkv_4d(h_ln)
    scaling = (640 // 20) ** -0.5
    qs = qs * scaling
    qm = qm * scaling
    # RoPE for self-attn only
    B_mb, H_mb, T_mb, D_mb = qs.shape
    q_rope = sa.rot_emb(qs.reshape(B_mb * H_mb, T_mb, D_mb))
    k_rope = sa.rot_emb(ks.reshape(B_mb * H_mb, T_mb, D_mb))
    qs_rope = q_rope.view(B_mb, H_mb, T_mb, D_mb).contiguous()
    ks_rope = k_rope.view(B_mb, H_mb, T_mb, D_mb).contiguous()
    
    def superfused():
        return fused_multi_pathway_attention(
            qs_rope, ks_rope, vss, qm, km, vmm, chain_mask,
            dropout_p=0.0, training=False)
    time_comp(superfused, "super-fused multi-pathway", iters=100)
    
    # Output projection
    def outproj():
        attn_out = superfused()
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, 640)
        sa.out_proj(attn_out)
    time_comp(outproj, "out_proj + reshape", iters=100)
    
    # FFN
    r2 = ln2(x)
    time_comp(lambda: ff(r2), "FFN (fc1+gelu+fc2)", iters=300)
    
    full_layer_ms = comps.get('super-fused multi-pathway', 0) + comps.get('out_proj + reshape', 0) + \
                    comps.get('self_attn QKV (3 GEMMs)', 0) + comps.get('multimer_attn fused QKV (1 GEMM)', 0) + \
                    comps.get('layernorm (both)', 0) + comps.get('FFN (fc1+gelu+fc2)', 0)
    
    print(f"\n  {'Component':<40s} {'ms':>8s} {'% of layer':>12s}")
    print(f"  {'-'*62}")
    for name, ms in sorted(comps.items(), key=lambda x: -x[1]):
        pct = ms / full_layer_ms * 100 if full_layer_ms > 0 else 0
        print(f"  {name:<40s} {ms:>8.3f} {pct:>11.1f}%")
    print(f"  {'-'*62}")
    print(f"  {'TOTAL (major components)':<40s} {full_layer_ms:>8.3f} {'100%':>12s}")
    
    # FLOPs estimation
    print(f"\n{'='*60}")
    print("FLOPs ESTIMATE (per layer, forward only)")
    print(f"{'='*60}")
    # QKV self: 3x (B*T*E*E*2) = 3 * 32768 * 640 * 640 * 2 = 80.5 GFLOPs
    # QKV multi: 1x (B*T*E*3E*2) = 32768 * 640 * 1920 * 2 = 80.5 GFLOPs
    # comb bmm: 2x (B*H*T*T*D*2) = 2 * 32*20*1024*1024*32*2 = 85.9 GFLOPs
    # weighted sum: 2x (B*H*T*T*D*2) = 85.9 GFLOPs
    # out_proj: B*T*E*E*2 = 26.8 GFLOPs
    # fc1: B*T*E*FF*2 = 32768 * 640 * 2560 * 2 = 107.4 GFLOPs
    # fc2: B*T*FF*E*2 = 32768 * 2560 * 640 * 2 = 107.4 GFLOPs
    flops_qkv = 80.5 + 80.5
    flops_attn = 85.9 + 85.9 + 26.8
    flops_ffn = 107.4 + 107.4
    flops_total = flops_qkv + flops_attn + flops_ffn
    print(f"  QKV projections:         {flops_qkv:.0f} GFLOPs ({flops_qkv/flops_total*100:.0f}%)")
    print(f"  Attention (bmm+sum+proj): {flops_attn:.0f} GFLOPs ({flops_attn/flops_total*100:.0f}%)")
    print(f"  FFN (fc1+fc2):           {flops_ffn:.0f} GFLOPs ({flops_ffn/flops_total*100:.0f}%)")
    print(f"  TOTAL per layer:         {flops_total:.0f} GFLOPs")
    print(f"  TOTAL 30 layers fwd:     {flops_total*30:.0f} GFLOPs ({flops_total*30/1000:.1f} TFLOPs)")


if __name__ == "__main__":
    profile()
