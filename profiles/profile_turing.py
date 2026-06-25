"""Profile MINT 150M on RTX 2080 Ti (CC 7.5, fp16, fallback pipeline)."""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint_plus.models.esm2 import MINT
from mint_plus.models.modules import build_checkpointed_model, enable_fused_multi_pathway
from mint_plus.models import MODEL_REGISTRY

torch.set_float32_matmul_precision('medium')

def time_block(fn, warmup=3, iters=10):
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

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)} CC: {torch.cuda.get_device_capability()}")
print(f"PyTorch: {torch.__version__}")
print()

config = MODEL_REGISTRY["150M"]
L, E, H = config["num_layers"], config["embed_dim"], config["attention_heads"]
D = E // H

model = MINT(num_layers=L, embed_dim=E, attention_heads=H, use_multimer=True)
ckpt = "./ckpts/esm2_150M/pytorch_model.bin"
if os.path.exists(ckpt):
    print(f"Loading pretrained weights from {ckpt}")
    model.load_pretrained_weights(ckpt, dtype=torch.float16)

# Freeze self-attn
model.requires_grad_(False)
for name, p in model.named_parameters():
    if 'multimer_attn' in name or 'lm' in name or 'norm' in name:
        p.requires_grad = True

# Block-3 checkpointing (NO super-fused kernel on Turing)
model = build_checkpointed_model(model, block_size=3)
model = model.cuda().to(torch.float16)
model.train()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
print()

for B in [8, 16, 24, 32]:
    for T in [512, 1024]:
        if B * T > 32 * 1024:
            continue
        tokens = torch.randint(4, 24, (B, T), device=device)
        tokens[:, 0] = 0
        tokens[:, -1] = 2
        chain_ids = torch.zeros(B, T, dtype=torch.int32, device=device)
        if T >= 1024:
            chain_ids[:, T//2:] = 1
        else:
            chain_ids[:, T//2:] = 1
        
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad()
        torch.cuda.synchronize()
        
        try:
            def step_fn():
                out = model(tokens, chain_ids=chain_ids)['logits']
                loss = out.sum()
                loss.backward()
                model.zero_grad()
            ms = time_block(step_fn, warmup=2, iters=5)
            peak = torch.cuda.max_memory_allocated() / 1e9
            tok_s = B * T * 1000 / ms
            print(f"  B={B:2d} T={T:4d}  step={ms:7.1f} ms  {tok_s:6.0f} tok/s  peak={peak:.2f} GB")
        except Exception as e:
            print(f"  B={B:2d} T={T:4d}  FAILED: {str(e)[:80]}")
