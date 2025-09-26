#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import pathlib

# Ensure local src is on path for in-repo execution
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

from lfm2_hybrid.model import LFM2_SSM_Small
from lfm2_hybrid.utils import set_seed


def parse_args():
    ap = argparse.ArgumentParser(description="POC inference for LFM2 hybrid")
    ap.add_argument("--prompt_ids", type=str, default="1,2,3,4")
    ap.add_argument("--vocab_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--attn_every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if np is None:
        raise SystemExit("NumPy required for POC inference. Please `pip install numpy`. ")

    ids = np.array([int(x) for x in args.prompt_ids.split(",")], dtype=np.int32)
    ids = ids[None, :]  # (1, T)
    model = LFM2_SSM_Small(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        attn_every=args.attn_every,
    )
    logits = model.forward(ids)
    next_id = int(np.argmax(logits[0, -1]))
    print(f"Next token id: {next_id}")


if __name__ == "__main__":
    main()
