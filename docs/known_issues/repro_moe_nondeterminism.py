#!/usr/bin/env python3
"""Minimal repro: aiter.fused_moe returns different results for identical input.

No server, no model weights -- one kernel call, repeated. Same tensors, same
pointers, same shapes, back to back on one GPU.

Context: serving Qwen3.5 at temperature 0 returns different text for identical
requests. Bisected by model, holding the whole stack fixed:

    Qwen3.5-27B      dense, bf16   ->  deterministic (logprob delta exactly 0)
    Qwen3.5-35B-A3B  MoE,   bf16   ->  nondeterministic (delta up to 2.1 nats)
    Qwen3.5-397B     MoE,   fp8    ->  nondeterministic (delta up to 1.6 nats)

so it is the MoE path, and not quantization. A single prefill forward over an
8-token prompt is already affected, so it is not accumulation over depth or
sequence length. `AITER_MOE_SMALL_BATCH=0` does not change it.

Usage: repro_moe_nondeterminism.py [--repeats 5] [--tokens 64] [--experts 256]
"""

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from aiter import ActivationType
    from aiter.fused_moe import fused_moe

    dtype = getattr(torch, args.dtype)
    dev = "cuda"
    torch.manual_seed(0)

    x = torch.randn(args.tokens, args.hidden, device=dev, dtype=dtype)
    w1 = torch.randn(
        args.experts, args.inter * 2, args.hidden, device=dev, dtype=dtype
    ) * 0.02
    w2 = torch.randn(args.experts, args.hidden, args.inter, device=dev, dtype=dtype) * 0.02

    gate = torch.randn(args.tokens, args.experts, device=dev, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(gate.softmax(-1), args.topk, dim=-1)
    topk_weight = topk_weight.to(torch.float32)  # kernel requires fp32
    topk_ids = topk_ids.to(torch.int32)

    outs = []
    for _ in range(args.repeats):
        out = fused_moe(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            expert_mask=None,
            activation=ActivationType.Silu,
        )
        outs.append(out.clone().float())

    base = outs[0]
    worst = max((o - base).abs().max().item() for o in outs[1:])
    n_diff = max(int((o != base).sum().item()) for o in outs[1:])
    bit_identical = sum(bool(torch.equal(o, base)) for o in outs[1:])
    scale = base.abs().max().item()
    print(
        f"tokens={args.tokens} experts={args.experts} topk={args.topk} "
        f"dtype={args.dtype}"
    )
    print(f"  repeats:                 {args.repeats}")
    print(f"  bit-identical to run 0:  {bit_identical}/{args.repeats - 1}")
    print(f"  elements differing:      {n_diff}/{base.numel()} "
          f"({100.0 * n_diff / base.numel():.1f}%)")
    print(f"  max abs difference:      {worst:.6e}")
    print(f"  output abs max:          {scale:.6e}")
    print(f"  relative:                {worst / scale:.3%}")

    # topk_ids are computed once and reused, so routing cannot be the variable:
    # any difference is downstream of expert selection.
    print("\n  (routing is fixed: topk_ids computed once, passed to every call)")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
