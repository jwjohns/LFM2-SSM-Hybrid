#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lfm2_hybrid.model import LFM2_SSM_Small  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Export model weights to NPZ + metadata JSON")
    ap.add_argument("--out", type=str, default="export/model.npz")
    ap.add_argument("--meta", type=str, default="export/meta.json")
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--attn_every", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=None)
    ap.add_argument("--parallel_residual", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = pathlib.Path(args.out)
    meta_path = pathlib.Path(args.meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

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
    sd = model.state_dict()

    # Save as NPZ with numpy arrays
    npz_dict = {k: v.detach().cpu().numpy() for k, v in sd.items()}
    np.savez(out_path, **npz_dict)

    # Minimal metadata to guide mapping in your llama.cpp importer
    meta = dict(
        arch="lfm2_hybrid",
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        attn_every=args.attn_every,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads or args.n_heads,
        parallel_residual=args.parallel_residual,
        rope_theta=args.rope_theta,
        notes="Names follow PyTorch state_dict; map to llama.cpp in your loader",
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Exported: {out_path} and {meta_path}")


if __name__ == "__main__":
    main()

