#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lfm2_hybrid.model import LFM2_SSM_Small  # noqa: E402


def md5_tensor(t: torch.Tensor) -> str:
    arr = t.detach().float().cpu().numpy().tobytes()
    return hashlib.md5(arr).hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description="Parity helper: print per-layer checksums and final logits")
    ap.add_argument("--tokens", type=str, default="1,2,3,4,5")
    ap.add_argument("--vocab_size", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--attn_every", type=int, default=1)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--parallel_residual", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    return ap.parse_args()


def main():
    args = parse_args()
    model = LFM2_SSM_Small(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        attn_every=args.attn_every,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        parallel_residual=args.parallel_residual,
        rope_theta=args.rope_theta,
    )
    model.eval()

    toks = torch.tensor([int(x) for x in args.tokens.split(",")], dtype=torch.long)[
        None, :
    ]

    # Hook outputs after each submodule to compare with your llama.cpp impl
    checks = {}

    def hook(name):
        def fn(module, inp, out):
            checks[name] = md5_tensor(out)

        return fn

    for i, m in enumerate(model.blocks):
        m.register_forward_hook(hook(f"block_{i}:{m.__class__.__name__}"))

    with torch.no_grad():
        logits = model(toks)

    result = {
        "config": {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "attn_every": args.attn_every,
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads,
            "parallel_residual": args.parallel_residual,
            "rope_theta": args.rope_theta,
        },
        "checksums": checks,
        "logits_md5": md5_tensor(logits),
        "last_logits": logits[0, -1].tolist(),
    }
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()

