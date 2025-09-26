#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import pathlib

# Ensure local src is on path for in-repo execution
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

from lfm2_hybrid.model import LFM2_SSM_Small
from lfm2_hybrid.utils import load_config, set_seed


def parse_args():
    ap = argparse.ArgumentParser(description="POC training step for LFM2 hybrid")
    ap.add_argument("--config", type=str, default="configs/small.json")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)

    vocab = int(cfg.get("vocab_size", 256))
    d = int(cfg.get("d_model", 128))
    n_layers = int(cfg.get("n_layers", 2))
    attn_every = int(cfg.get("attn_every", 2))
    batch = int(cfg.get("batch_size", 2))
    seq = int(cfg.get("seq_len", 16))

    if np is None:
        raise SystemExit("NumPy required for POC training. Please `pip install numpy`. ")

    model = LFM2_SSM_Small(vocab, d, n_layers, attn_every=attn_every)
    # Fake data and one step forward to validate shapes
    token_ids = np.random.randint(0, vocab, size=(batch, seq), dtype=np.int32)
    logits = model.forward(token_ids)

    print(
        f"OK: logits shape {logits.shape} for (batch={batch}, seq={seq}, vocab={vocab})"
    )


if __name__ == "__main__":
    main()
