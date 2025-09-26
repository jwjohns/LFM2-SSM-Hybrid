#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Export a Hugging Face model's state_dict to NPZ + metadata JSON")
    ap.add_argument("--model", type=str, default="LiquidAI/LFM2-1.2B")
    ap.add_argument("--out", type=str, default="export/lfm2_1p2b.npz")
    ap.add_argument("--meta", type=str, default="export/lfm2_1p2b_meta.json")
    ap.add_argument("--local", action="store_true", help="Use local cache only (no download)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    meta = pathlib.Path(args.meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except Exception as e:
        raise SystemExit("Please install transformers: pip install transformers")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=args.local, trust_remote_code=True
    )
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.local, trust_remote_code=True)

    sd = model.state_dict()
    npz_dict = {k: v.cpu().numpy() for k, v in sd.items()}
    np.savez(out, **npz_dict)

    meta.write_text(
        json.dumps(
            {
                "hf_model": args.model,
                "config": cfg.to_dict(),
                "num_tensors": len(sd),
            },
            indent=2,
        )
    )
    print(f"Exported NPZ to {out} and meta to {meta}")


if __name__ == "__main__":
    main()

