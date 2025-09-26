#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, Tuple, Optional

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lfm2_hybrid.model import LFM2_SSM_Small  # noqa: E402


def load_npz(path: str) -> Dict[str, torch.Tensor]:
    import numpy as np

    data = np.load(path)
    return {k: torch.from_numpy(data[k]) for k in data.files}


def load_torch(path: str) -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu")
    # Allow either direct state_dict or {"model": state_dict}
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    if isinstance(sd, dict):
        return sd
    raise ValueError("Unsupported checkpoint format")


def match_and_load(model: torch.nn.Module, src: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    dst = model.state_dict()
    loaded = 0
    skipped = 0
    new_sd = {}
    for k, v in dst.items():
        if k in src and tuple(src[k].shape) == tuple(v.shape):
            new_sd[k] = src[k]
            loaded += 1
        else:
            new_sd[k] = v
            skipped += 1
    model.load_state_dict(new_sd)
    return loaded, skipped


def _infer_dims(src: Dict[str, torch.Tensor]) -> Tuple[Optional[int], Optional[int]]:
    # Try to find an embedding-like matrix (vocab, d_model)
    candidates = []
    for k, v in src.items():
        if v.ndim == 2:
            V, D = v.shape
            score = 0
            name = k.lower()
            if any(s in name for s in ["embed", "tok", "wte", "embedding"]):
                score += 10
            if V >= D:
                score += 1
            candidates.append((score, V, D, k))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _, V, D, key = candidates[0]
    return int(V), int(D)


def parse_args():
    ap = argparse.ArgumentParser(description="Partially load weights into our model by matching names and shapes")
    ap.add_argument("--src", type=str, required=True, help="Source weights: .npz or .pt/.pth")
    ap.add_argument("--save", type=str, required=True, help="Output checkpoint path (.pt)")
    ap.add_argument("--vocab_size", type=int, default=0, help="0=auto from src if possible")
    ap.add_argument("--d_model", type=int, default=0, help="0=auto from src if possible")
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--attn_every", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=None)
    ap.add_argument("--parallel_residual", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    return ap.parse_args()


def main():
    args = parse_args()
    # Optionally infer dims from source
    if args.src.endswith(".npz"):
        src = load_npz(args.src)
    else:
        src = load_torch(args.src)

    vocab, d = _infer_dims(src)
    vocab_size = args.vocab_size or (vocab or 256)
    d_model = args.d_model or (d or 256)

    model = LFM2_SSM_Small(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=args.n_layers,
        attn_every=args.attn_every,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        parallel_residual=args.parallel_residual,
        rope_theta=args.rope_theta,
    )

    loaded, skipped = match_and_load(model, src)
    torch.save({"model": model.state_dict(), "config": {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": args.n_layers,
        "attn_every": args.attn_every,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "parallel_residual": args.parallel_residual,
        "rope_theta": args.rope_theta,
    }}, args.save)
    print(f"Loaded {loaded} tensors, skipped {skipped}. Inferred vocab={vocab_size}, d_model={d_model}. Saved to {args.save}")


if __name__ == "__main__":
    main()
