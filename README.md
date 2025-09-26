# LFM2-SSM-Hybrid

Setup, Init, Train, Export

This repo contains a PyTorch implementation of a hybrid MLP + SSM decoder with Nemotron‑style options (RoPE, grouped‑query attention, parallel residual). It includes scripts to initialize from LiquidAI/LFM2‑1.2B in your local Hugging Face cache, train on text, generate, export weights, and run parity checks for a llama.cpp port.

## Prerequisites (UV only)
- Python 3.10+
- We use UV exclusively for Python management: https://docs.astral.sh/uv/
- Create env and install tooling and deps:
  - `uv venv`
  - `source .venv/bin/activate`
  - `uv pip install -e .[dev]`
  - `uv pip install torch transformers`

Directory highlights
- `src/lfm2_hybrid/`: model and blocks (RMSNorm, GatedMLP, DiagonalSSM, RoPE + GQA attention)
- `scripts/`: CLI utilities (init, train, infer, export, parity)
- `configs/`: minimal JSON example
- `tests/`: torch shape test

## Initialize From Local HF Cache (LFM2‑1.2B)
Creates a checkpoint of our hybrid model seeded with compatible LFM2 tensors (embeddings, final norm, lm_head) and matching dims.

- Command:
  - `python scripts/init_from_hf.py --model LiquidAI/LFM2-1.2B --local --out ckpt_from_hf.pt --attn_every 2 --parallel_residual`
- Notes:
  - `--local` uses only local Hugging Face cache (no network).
  - Dims (`vocab_size`, `d_model`, `n_layers`, `n_heads`) are inferred from the HF config.
  - Use `--n_kv_heads` if you want grouped‑query (must divide `n_heads`).
  - You can adjust `--attn_every` and `--rope_theta` to your target.

Alternative: export NPZ from HF cache, then load partially.
- Export: `python scripts/hf_export_npz.py --model LiquidAI/LFM2-1.2B --local --out export/lfm2_1p2b.npz --meta export/lfm2_meta.json`
- Partial load: `python scripts/partial_load.py --src export/lfm2_1p2b.npz --save ckpt_from_lfm2.pt`
  - Auto‑infers `vocab_size` and `d_model` from source tensors and loads all matching name/shape pairs.

## Train
Train next‑token LM with teacher forcing. Supports tokenizer‑based or byte‑level inputs.

- Tokenizer‑based (recommended):
  - `python scripts/train_torch.py \
      --device cuda \
      --data_path path/to/text.txt \
      --hf_tokenizer LiquidAI/LFM2-1.2B \
      --init_from ckpt_from_hf.pt \
      --save_path ckpt.pt \
      --n_kv_heads 4 --parallel_residual \
      --steps 200 --batch_size 4 --seq_len 128 --lr 3e-4`
- Byte‑level fallback (no tokenizer):
  - `python scripts/train_torch.py --device cpu --data_path path/to/text.txt --steps 200 --save_path ckpt.pt`

Important flags
- `--init_from`: NPZ/PT to partially load before training (e.g., HF export or `ckpt_from_hf.pt`).
- `--hf_tokenizer`: Hugging Face tokenizer name/path to tokenize text (keeps vocab aligned with LFM2).
- `--n_kv_heads`: grouped‑query attention (must divide `n_heads`).
- `--parallel_residual`: use parallel MLP+SSM residual mixing.
- `--rope_theta`: RoPE base (default 10000.0).

## Inference
Generate greedily from a checkpoint or a partially initialized model.

- From a trained checkpoint:
  - `python scripts/infer_torch.py --load_path ckpt.pt --hf_tokenizer LiquidAI/LFM2-1.2B --prompt "hello world" --max_new_tokens 32 --n_kv_heads 4 --parallel_residual`
- From partial init (HF export) without training:
  - `python scripts/infer_torch.py --init_from export/lfm2_1p2b.npz --hf_tokenizer LiquidAI/LFM2-1.2B --prompt "hello world" --max_new_tokens 16 --n_kv_heads 4 --parallel_residual`
- Byte‑level prompt (no tokenizer):
  - `python scripts/infer_torch.py --prompt_ids 1,2,3,4 --max_new_tokens 8`

## Export + Parity
- Export our model to NPZ + meta JSON:
  - `python scripts/export_npz.py --out export/model.npz --meta export/meta.json --n_kv_heads 4 --parallel_residual`
- HF export (local cache):
  - `python scripts/hf_export_npz.py --model LiquidAI/LFM2-1.2B --local --out export/lfm2_1p2b.npz`
- Parity helper (compare with your C++ forward):
  - `python scripts/parity_check.py --tokens 1,2,3,4 --n_kv_heads 4 --parallel_residual`
  - Prints per‑layer checksums and final logits.

## Nemotron/Llama.cpp Mapping Tips
- RoPE: apply to Q,K with same `rope_theta`; `head_dim` must be even.
- GQA: ensure `n_heads % n_kv_heads == 0`; repeat K,V per head group.
- Names: exporter uses PyTorch `state_dict` keys; read `meta.json` and map to your llama.cpp schema (e.g., `attn_q/k/v/o`, `ffn_gate/up/down`, `attention_norm`, `ffn_norm`, `tok_embeddings`, `output`).
- Parallel residual: sum SSM and MLP branches off the same input; match your C++ topology for parity.

## Makefile Shortcuts (UV-backed)
- `make setup` → create `.venv` and install dev + torch + transformers via UV
- `make train` → quick CPU training demo (UV run)
- `make infer` → simple greedy generation (UV run)
- `make export` → export NPZ (UV run)
- `make parity` → parity helper (UV run)

## Troubleshooting
- Torch/Transformers missing: `pip install torch transformers`.
- HF local cache only: pass `--local` to avoid downloads. Ensure the model is present (`LiquidAI/LFM2-1.2B`).
- Tokenizer mismatch: specify `--hf_tokenizer LiquidAI/LFM2-1.2B` for LFM2 text.
- Dimension mismatch: use `init_from_hf.py` to infer dims from config; for `partial_load.py`, pass explicit `--vocab_size/--d_model` if auto‑infer fails.
- GQA errors: make sure `--n_heads % --n_kv_heads == 0`.
- CUDA issues: switch to `--device cpu` to debug, then back to CUDA.

## Notes & Next Steps
- Current HF → hybrid mapping copies embeddings, final norm, and lm_head. If you want deeper 1:1 initialization, provide exact HF module names per layer and we’ll add a converter.
- For datasets beyond a single file, wire a loader (HF Datasets) and curriculum; scripts are structured to extend easily.
