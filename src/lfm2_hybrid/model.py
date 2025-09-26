from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SSMBlock, AttnBlock, RMSNorm, HybridBlockParallel


class LFM2_SSM_Small(nn.Module):
    """Decoder-only stack with SSM and sparse attention blocks (PyTorch)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        attn_every: int | None = 4,
        n_heads: int = 4,
        n_kv_heads: int | None = None,
        mlp_mult: float = 4.0,
        parallel_residual: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_every = attn_every

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(4096, d_model)  # learned positions

        blocks = []
        for i in range(n_layers):
            if parallel_residual:
                blocks.append(HybridBlockParallel(d_model, mlp_mult=mlp_mult))
            else:
                blocks.append(SSMBlock(d_model, mlp_mult=mlp_mult))
            if attn_every and (i + 1) % attn_every == 0:
                blocks.append(
                    AttnBlock(
                        d_model,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        rope_theta=rope_theta,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) long
        B, T = token_ids.shape
        x = self.embed(token_ids) + self.pos(torch.arange(T, device=token_ids.device))
        for b in self.blocks:
            x = b(x)
        x = self.out_norm(x)
        logits = self.lm_head(x)
        return logits
