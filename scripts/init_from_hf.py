#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict

import torch


def parse_args():
    ap = argparse.ArgumentParser(description="Initialize our hybrid model from a local HF model (cache) and save a checkpoint")
    ap.add_argument("--model", type=str, default="LiquidAI/LFM2-1.2B")
    ap.add_argument("--out", type=str, default="ckpt_from_hf.pt")
    ap.add_argument("--local", action="store_true", help="Use local HF cache only (no download)")
    ap.add_argument("--attn_every", type=int, default=2)
    ap.add_argument("--n_kv_heads", type=int, default=None)
    ap.add_argument("--parallel_residual", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except Exception:
        raise SystemExit("Please install transformers: pip install transformers")

    # Load config and model from local cache
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.local, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=args.local, trust_remote_code=True
    )
    sd = hf_model.state_dict()

    # Infer dims
    vocab_size = getattr(cfg, "vocab_size", None) or getattr(cfg, "n_vocab", None)
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    n_kv_heads = args.n_kv_heads or getattr(cfg, "num_key_value_heads", None)
    if any(x is None for x in [vocab_size, d_model, n_layers, n_heads]):
        raise SystemExit("Could not infer model dims from HF config; please specify manually.")

    # Import here to avoid import if transformers missing
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from lfm2_hybrid.model import LFM2_SSM_Small  # noqa: E402

    model = LFM2_SSM_Small(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_layers=int(n_layers),
        attn_every=int(args.attn_every),
        n_heads=int(n_heads),
        n_kv_heads=None if n_kv_heads is None else int(n_kv_heads),
        parallel_residual=bool(args.parallel_residual),
        rope_theta=float(args.rope_theta),
    )

    # Map HF weights into our model where it makes sense
    ours = model.state_dict()
    mapped: Dict[str, torch.Tensor] = {}

    def try_copy(src_key: str, dst_key: str):
        if src_key in sd and dst_key in ours and tuple(sd[src_key].shape) == tuple(ours[dst_key].shape):
            mapped[dst_key] = sd[src_key]

    # Common names
    try_copy("model.embed_tokens.weight", "embed.weight")
    try_copy("transformer.wte.weight", "embed.weight")
    try_copy("lm_head.weight", "lm_head.weight")
    try_copy("model.norm.weight", "out_norm.weight")
    try_copy("transformer.ln_f.weight", "out_norm.weight")

    # Apply mapped values
    for k, v in mapped.items():
        ours[k] = v
    model.load_state_dict(ours)

    ckpt = {
        "model": model.state_dict(),
        "config": {
            "vocab_size": int(vocab_size),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "n_kv_heads": None if n_kv_heads is None else int(n_kv_heads),
            "attn_every": int(args.attn_every),
            "parallel_residual": bool(args.parallel_residual),
            "rope_theta": float(args.rope_theta),
            "hf_model": args.model,
            "mapped": sorted(list(mapped.keys())),
        },
    }
    torch.save(ckpt, args.out)
    print(f"Initialized from HF cache: {args.model}\nSaved checkpoint to {args.out}\nMapped tensors: {list(mapped.keys())}")


if __name__ == "__main__":
    main()

