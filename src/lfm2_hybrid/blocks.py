"""Real PyTorch blocks for the LFM2 hybrid POC.

Includes:
- RMSNorm
- Diagonal SSM (simple recurrent linear state-space per channel)
- Gated MLP (SwiGLU)
- Multi-head self-attention (causal)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


class GatedMLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=True)
        self.w2 = nn.Linear(d, hidden, bias=True)
        self.w3 = nn.Linear(hidden, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w2(x)) * self.w1(x))


class DiagonalSSM(nn.Module):
    """A simple diagonal state-space layer.

    y_t[c] = a[c] * y_{t-1}[c] + b[c] * x_t[c]
    """

    def __init__(self, d: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(d))  # initialized near 0
        self.b = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, D = x.shape
        a = torch.tanh(self.a)  # keep stable, in (-1,1)
        b = self.b
        y = torch.zeros_like(x)
        prev = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(T):
            prev = a * prev + b * x[:, t, :]
            y[:, t, :] = prev
        return self.out(y)


class SSMBlock(nn.Module):
    def __init__(self, d: int, mlp_mult: float = 4.0):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.ssm = DiagonalSSM(d)
        self.n2 = RMSNorm(d)
        self.mlp = GatedMLP(d, int(d * mlp_mult))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, heads, T, hd)
    B, H, T, D = x.shape
    x_ = x.view(B, H, T, D // 2, 2)
    x1, x2 = x_[..., 0], x_[..., 1]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).reshape(B, H, T, D)


def _rope_cache(
    T: int, hd: int, device, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    freqs = torch.einsum("t,d->td", t, theta)  # (T, hd/2)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,T,hd/2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
    return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be multiple of n_kv_heads"
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.hd = d // n_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, HKV, HD = self.n_heads, self.n_kv_heads, self.hd
        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)  # (B,H,T,HD)
        k = self.k_proj(x).view(B, T, HKV, HD).transpose(1, 2)  # (B,HKV,T,HD)
        v = self.v_proj(x).view(B, T, HKV, HD).transpose(1, 2)  # (B,HKV,T,HD)

        # RoPE on q,k
        cos, sin = _rope_cache(T, HD, x.device, base=self.rope_theta)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: repeat k,v along head groups
        if H != HKV:
            repeat = H // HKV
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product with causal mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HD)  # (B,H,T,T)
        mask = torch.full((T, T), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        attn_scores = attn_scores + mask
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.drop(attn)
        y = torch.matmul(attn, v)  # (B,H,T,HD)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.o_proj(y)
        return y


class AttnBlock(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.n = RMSNorm(d)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.attn = CausalSelfAttention(
            d,
            n_heads=n_heads,
            n_kv_heads=self.n_kv_heads,
            dropout=dropout,
            rope_theta=rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n(x)
        out = self.attn(h)
        return x + out


class HybridBlockParallel(nn.Module):
    """Parallel residual: SSM and MLP branches off the same input.

    y = x + SSM(norm1(x)) + MLP(norm2(x))
    """

    def __init__(self, d: int, mlp_mult: float = 4.0):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.ssm = DiagonalSSM(d)
        self.mlp = GatedMLP(d, int(d * mlp_mult))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.ssm(self.n1(x))
        h2 = self.mlp(self.n2(x))
        return x + h1 + h2
