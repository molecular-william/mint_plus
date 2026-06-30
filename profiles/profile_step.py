"""Profile MINT 150M frozen training step with per-component breakdown.

Usage:
    CUDA_VISIBLE_DEVICES=0 python profiles/profile_step.py
"""

import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint_plus.models.esm2 import MINT
from mint_plus.models.modules import build_checkpointed_model, TransformerLayer_MINT
from mint_plus.data.data import STRINGDataset, CollateFn
from mint_plus.models import MODEL_REGISTRY


def time_block(fn, warmup=5, iters=50, name="block"):
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
    """Build the model exactly like the trainer does, but without compile."""
    config = MODEL_REGISTRY["150M"]
    model = MINT(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        use_multimer=True,
    )
    # Load pretrained weights
    ckpt = "./ckpts/esm2_150M/pytorch_model.bin"
    if os.path.exists(ckpt):
        print(f"Loading pretrained weights from {ckpt}")
        model.load_pretrained_weights(ckpt, dtype=torch.bfloat16)
    
    # Build checkpointed blocks (block_size=3)
    model = build_checkpointed_model(model, block_size=3)
    model = model.cuda().to(torch.bfloat16)
    model.train()
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
    return model


def get_data_loader(batch_size=32, crop_length=512, max_examples=5000):
    collate_fn = CollateFn(crop_length)
    ds = STRINGDataset(
        links_path="data/diamond/validation.links.txt.zst",
        seqs_path="data/diamond/validation.seqs.txt.zst",
        global_rank=0, world_size=1, max_examples=max_examples
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=collate_fn,
        pin_memory=True, num_workers=0,
    )
    return loader


def profile():
    device = "cuda"
    torch.set_float32_matmul_precision('medium')
    
    cfg = MODEL_REGISTRY["150M"]
    L, E, H, FF = cfg["num_layers"], cfg["embed_dim"], cfg["attention_heads"], cfg["intermediate_size"]
    D = E // H
    print(f"Architecture: L={L} layers, E={E}, H={H}, D={D}, FF={FF}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    
    # ---- Build model ----
    model = build_model()
    
    # ---- Get a batch ----
    loader = get_data_loader(batch_size=32)
    tokens, chain_ids = next(iter(loader))
    T = tokens.shape[1]
    print(f"Sample batch: ({tokens.shape[0]}, {T}) -- actual T = 2 * crop_length = {T}")
    
    # ---- Profile at batch sizes ----
    for B in [16, 32, 48]:
        print(f"\n{'='*60}")
        print(f"BATCH SIZE B={B}")
        print(f"{'='*60}")
        
        t = tokens[:B].cuda()
        c = chain_ids[:B].cuda()
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        def step():
            out = model(t, chain_ids=c)
            loss = out["logits"].sum()
            loss.backward()
        
        try:
            ms_fwd_bwd = time_block(step, warmup=3, iters=10, name=f"fwd_bwd_B={B}")
            peak = torch.cuda.max_memory_allocated() / 1e9
            tok_s = B * T * 1000 / ms_fwd_bwd
            print(f"  fwd+bwd: {ms_fwd_bwd:.1f} ms, {tok_s:.0f} tok/s, peak={peak:.2f} GB")
        except RuntimeError as e:
            print(f"  FAILED: {str(e)[:120]}")
            continue
        
        # Forward-only
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        def fwd_only():
            out = model(t, chain_ids=c)
            out["logits"].sum()
        
        try:
            ms_fwd = time_block(fwd_only, warmup=3, iters=20, name=f"fwd_B={B}")
            print(f"  forward:   {ms_fwd:.1f} ms ({ms_fwd/ms_fwd_bwd*100:.0f}%)")
            print(f"  backward:  {ms_fwd_bwd-ms_fwd:.1f} ms ({(ms_fwd_bwd-ms_fwd)/ms_fwd_bwd*100:.0f}%)")
        except:
            pass
    
    # ---- Per-layer micro-benchmark at B=32 ----
    print(f"\n{'='*60}")
    print(f"PER-LAYER MICRO-BENCHMARK (B=32)")
    print(f"{'='*60}")
    
    B = 32
    t = tokens[:B].cuda()
    c = chain_ids[:B].cuda()
    T = t.shape[1]
    
    # Extract first layer
    block0 = model.layers[0]
    layer = block0.layers[0]
    
    print(f"  Layer: {type(layer).__name__}")
    
    x = torch.randn(T, B, E, device=device, dtype=torch.bfloat16, requires_grad=True)
    padding_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
    chain_mask = ~torch.eq(c.unsqueeze(-1), c.unsqueeze(-2))
    
    sa = layer.self_attn
    ma = layer.multimer_attn
    ff = layer.feed_forward
    ln1 = layer.self_attn_layer_norm
    ln2 = layer.final_layer_norm
    
    # Full layer timing
    def full_layer():
        r = x
        h = ln1(r)
        # _multimer_attention_plus pipeline
        intra_logits, intra_v = sa(x=h, key_padding_mask=padding_mask, before_softmax=True)
        inter_logits, inter_v = ma(x=h, key_padding_mask=padding_mask, before_softmax=True)
        from mint_plus.models.kernels import fused_multimer_combine
        attn_out = fused_multimer_combine(intra_logits, inter_logits, chain_mask, intra_v, inter_v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        h2 = sa.out_proj(attn_out).transpose(0, 1).contiguous()
        h2 = r + h2
        r2 = h2
        h2 = ln2(h2)
        h2 = ff(h2)
        h2 = r2 + h2
        return h2
    
    ms_full = time_block(full_layer, warmup=10, iters=100, name="full_layer")
    
    # Per-component
    comps = {}
    from mint_plus.models.kernels import fused_multimer_combine
    
    def time_comp(fn, warmup=10, iters=200, name=""):
        comps[name] = time_block(fn, warmup=warmup, iters=iters, name=name)
    
    time_comp(lambda: (ln1(x), ln2(x)), name="layernorm (both)", iters=500)
    
    h_ln = ln1(x)
    time_comp(lambda: (sa.q_proj(h_ln), sa.k_proj(h_ln), sa.v_proj(h_ln)), 
              name="self_attn QKV proj", iters=300)
    time_comp(lambda: (ma.q_proj(h_ln), ma.k_proj(h_ln), ma.v_proj(h_ln)),
              name="multimer_attn QKV proj", iters=300)
    
    # Self-attn bmm (with rotary)
    def self_bmm():
        q = sa.q_proj(h_ln) * sa.scaling
        k = sa.k_proj(h_ln)
        q = q.view(T, B*H, D).transpose(0, 1)
        k = k.view(T, B*H, D).transpose(0, 1)
        q, k = sa.rot_emb(q), sa.rot_emb(k)
        return torch.bmm(q, k.transpose(1, 2))
    time_comp(self_bmm, name="self_attn bmm (QK^T) + RoPE", iters=100)
    
    # Multimer bmm
    def multi_bmm():
        q = ma.q_proj(h_ln) * ma.scaling
        k = ma.k_proj(h_ln)
        q = q.view(T, B*H, D).transpose(0, 1)
        k = k.view(T, B*H, D).transpose(0, 1)
        return torch.bmm(q, k.transpose(1, 2))
    time_comp(multi_bmm, name="multimer_attn bmm (QK^T)", iters=100)
    
    # Fused multimer combine (the whole pipeline)
    def fused_combine():
        intra_l, intra_v = sa(x=h_ln, key_padding_mask=padding_mask, before_softmax=True)
        inter_l, inter_v = ma(x=h_ln, key_padding_mask=padding_mask, before_softmax=True)
        return fused_multimer_combine(intra_l, inter_l, chain_mask, intra_v, inter_v, dropout_p=0.0)
    time_comp(fused_combine, name="fused_multimer_combine (full)", iters=100)
    
    # Output projection
    def outproj():
        attn_out = fused_combine()
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        sa.out_proj(attn_out)
    time_comp(outproj, name="out_proj + reshape", iters=100)
    
    # FFN
    r2 = ln2(x)
    time_comp(lambda: ff(r2), name="FFN (fc1+gelu+fc2)", iters=300)
    
    print(f"\n  {'Component':<35s} {'ms':>8s} {'% of layer':>12s}")
    print(f"  {'-'*57}")
    for name, ms in sorted(comps.items(), key=lambda x: -x[1]):
        print(f"  {name:<35s} {ms:>8.3f} {ms/ms_full*100:>11.1f}%")
    print(f"  {'-'*57}")
    print(f"  {'full_layer (fwd only)':<35s} {ms_full:>8.3f} {'100.0%':>12s}")
    
    # ---- Data loading cost ----
    print(f"\n{'='*60}")
    print(f"DATA LOADING COST")
    print(f"{'='*60}")
    data_loader = get_data_loader(batch_size=32, max_examples=200)
    dl_ms = time_block(lambda: next(iter(data_loader)), warmup=3, iters=10, name="dataload")
    print(f"  Single fetch: {dl_ms:.1f} ms")
    
    # ---- Memory breakdown ----
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"MEMORY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Model params (bf16):          {total*2/1e9:.2f} GB")
    print(f"  Gradients (bf16, trainable):  {trainable*2/1e9:.2f} GB")
    print(f"  Adam states (fp32, trainable): {trainable*4*2/1e9:.2f} GB")
    print(f"  Base footprint:               {(total*2 + trainable*2 + trainable*8)/1e9:.2f} GB")


if __name__ == "__main__":
    profile()
