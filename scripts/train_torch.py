#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
import torch.optim as optim

from lfm2_hybrid.model import LFM2_SSM_Small


def parse_args():
    ap = argparse.ArgumentParser(description="Train step for LFM2 hybrid (PyTorch)")
    ap.add_argument("--vocab_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--attn_every", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=None)
    ap.add_argument("--parallel_residual", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--data_path", type=str, default=None, help="Optional path to .txt for training")
    ap.add_argument("--save_path", type=str, default=None, help="Optional path to save checkpoint .pt")
    ap.add_argument("--init_from", type=str, default=None, help="Optional NPZ/.pt to partially load before training")
    ap.add_argument("--hf_tokenizer", type=str, default=None, help="Optional HF tokenizer name or path for text tokenization")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

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

    # Optional partial initialization
    if args.init_from:
        if args.init_from.endswith('.npz'):
            from partial_load import load_npz, match_and_load  # type: ignore
            src = load_npz(args.init_from)
        else:
            from partial_load import load_torch, match_and_load  # type: ignore
            src = load_torch(args.init_from)
        loaded, skipped = match_and_load(model, src)
        print(f"init_from loaded={loaded} skipped={skipped}")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    data_ids: Optional[torch.Tensor] = None
    if args.data_path:
        if args.hf_tokenizer:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, local_files_only=True, trust_remote_code=True)
                text = pathlib.Path(args.data_path).read_text(encoding='utf-8')
                ids = tok.encode(text, add_special_tokens=False)
                data_ids = torch.tensor(ids, dtype=torch.long, device=device)
                if args.vocab_size != tok.vocab_size:
                    print(f"[warn] vocab_size {args.vocab_size} != tokenizer.vocab_size {tok.vocab_size}")
            except Exception as e:
                raise SystemExit("Install transformers or omit --hf_tokenizer to use byte-level.")
        else:
            text = pathlib.Path(args.data_path).read_bytes()
            data_ids = torch.tensor(list(text), dtype=torch.long, device=device)

    def sample_batch():
        if data_ids is None or data_ids.numel() < args.seq_len + 1:
            toks = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len + 1), device=device)
        else:
            toks = torch.empty((args.batch_size, args.seq_len + 1), dtype=torch.long, device=device)
            max_start = data_ids.numel() - (args.seq_len + 1)
            for b in range(args.batch_size):
                s = torch.randint(0, max_start + 1, (1,), device=device).item()
                toks[b] = data_ids[s:s + args.seq_len + 1]
        return toks[:, :-1], toks[:, 1:]

    for step in range(1, args.steps + 1):
        tokens, targets = sample_batch()
        logits = model(tokens)  # (B, T, V)
        loss = loss_fn(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")

    if args.save_path:
        ckpt = {
            "model": model.state_dict(),
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
        }
        torch.save(ckpt, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
