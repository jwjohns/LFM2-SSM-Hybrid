#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from lfm2_hybrid.model import LFM2_SSM_Small


def parse_args():
    ap = argparse.ArgumentParser(description="Greedy generation (PyTorch)")
    ap.add_argument("--prompt_ids", type=str, default="1,2,3,4")
    ap.add_argument("--prompt", type=str, default=None, help="Optional text prompt (byte-level)")
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--attn_every", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=None)
    ap.add_argument("--parallel_residual", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--load_path", type=str, default=None, help="Optional checkpoint .pt to load")
    ap.add_argument("--init_from", type=str, default=None, help="Optional NPZ/.pt to partially load before generation")
    ap.add_argument("--hf_tokenizer", type=str, default=None, help="Optional HF tokenizer name or path for text prompts")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.prompt is not None and args.hf_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, local_files_only=True, trust_remote_code=True)
            toks = tok.encode(args.prompt, add_special_tokens=False)
            tokens = torch.tensor(toks, dtype=torch.long)[None, :].to(device)
        except Exception as e:
            raise SystemExit("Install transformers or remove --hf_tokenizer to use byte-level prompts.")
    elif args.prompt is not None:
        byte_ids = list(args.prompt.encode("utf-8"))
        tokens = torch.tensor(byte_ids, dtype=torch.long)[None, :].to(device)
    else:
        tokens = torch.tensor([int(x) for x in args.prompt_ids.split(",")], dtype=torch.long)[
            None, :
        ].to(device)

    model = LFM2_SSM_Small(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        attn_every=args.attn_every,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        parallel_residual=args.parallel_residual,
        rope_theta=args.rope_theta,
    ).to(device)
    if args.init_from:
        # Partial init from NPZ/.pt
        from partial_load import load_npz, load_torch, match_and_load  # type: ignore
        if args.init_from.endswith('.npz'):
            src = load_npz(args.init_from)
        else:
            src = load_torch(args.init_from)
        match_and_load(model, src)

    if args.load_path:
        ckpt = torch.load(args.load_path, map_location=device)
        model.load_state_dict(ckpt["model"]) 
    model.eval()

    for _ in range(args.max_new_tokens):
        logits = model(tokens)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_id], dim=1)

    if args.prompt is not None and args.hf_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, local_files_only=True, trust_remote_code=True)
            print(tok.decode(tokens.squeeze(0).tolist()))
        except Exception:
            print(tokens.squeeze(0).tolist())
    elif args.prompt is not None:
        try:
            print(bytes(tokens.squeeze(0).tolist()).decode("utf-8", errors="ignore"))
        except Exception:
            print(tokens.squeeze(0).tolist())
    else:
        print(tokens.squeeze(0).tolist())


if __name__ == "__main__":
    main()
