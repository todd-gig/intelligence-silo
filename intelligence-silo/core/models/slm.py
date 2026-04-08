"""Small Language Model — lightweight transformer micro-expert for local inference."""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelType(Enum):
    ENCODER = "encoder"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    REGRESSION = "regression"
    GENERATIVE = "generative"


@dataclass
class SLMConfig:
    """Configuration for a single small language model."""
    name: str
    model_type: ModelType
    vocab_size: int = 8192
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int | None = None  # defaults to 4 * hidden_dim
    dropout: float = 0.1
    max_seq_len: int = 512
    num_classes: int = 32  # for classification
    output_dim: int = 1  # for regression
    role: str = ""

    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_dim


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for position-aware attention."""

    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos[..., :d] - x2 * sin[..., :d],
                      x2 * cos[..., :d] + x1 * sin[..., :d]], dim=-1)


class SLMAttention(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(T)
        cos = cos[:T].unsqueeze(0).unsqueeze(0)
        sin = sin[:T].unsqueeze(0).unsqueeze(0)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class SLMBlock(nn.Module):
    """Transformer block with pre-norm, RoPE attention, and SwiGLU FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = SLMAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.norm2 = nn.RMSNorm(hidden_dim)
        # SwiGLU FFN
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm attention
        h = x + self.attn(self.norm1(x), mask)
        # SwiGLU FFN
        normed = self.norm2(h)
        ffn = self.w2(F.silu(self.w1(normed)) * self.w3(normed))
        return h + self.dropout(ffn)


class SmallLanguageModel(nn.Module):
    """Lightweight transformer micro-expert.

    Each SLM is a specialized model within the matrix:
    - Classifier: maps inputs to decision categories
    - Scorer: predicts value/penalty dimensions
    - Trust Assessor: evaluates trust tiers
    - Memory Encoder: generates memory embeddings
    - Pattern Detector: identifies recurring patterns
    - Causal Predictor: forecasts downstream effects

    Architecture: RoPE + RMSNorm + SwiGLU (modern LLM stack, scaled down).
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config

        # Token + positional embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SLMBlock(
                config.hidden_dim, config.num_heads, config.ff_dim,
                config.dropout, config.max_seq_len,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.RMSNorm(config.hidden_dim)

        # Task-specific head
        if config.model_type == ModelType.SEQUENCE_CLASSIFICATION:
            self.head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_classes),
            )
        elif config.model_type == ModelType.REGRESSION:
            self.head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.output_dim),
            )
        elif config.model_type == ModelType.ENCODER:
            self.head = nn.Linear(config.hidden_dim, config.hidden_dim)
        elif config.model_type == ModelType.GENERATIVE:
            self.head = nn.Linear(config.hidden_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token indices
            mask: optional attention mask

        Returns:
            Task-dependent output tensor.
        """
        x = self.embed_dropout(self.token_embed(input_ids))

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        if self.config.model_type in (ModelType.SEQUENCE_CLASSIFICATION, ModelType.REGRESSION):
            # Pool: mean over sequence
            x = x.mean(dim=1)
        elif self.config.model_type == ModelType.ENCODER:
            x = x.mean(dim=1)
        # GENERATIVE: return full sequence logits

        return self.head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_human(self) -> str:
        count = self.param_count()
        if count >= 1e6:
            return f"{count / 1e6:.1f}M"
        return f"{count / 1e3:.0f}K"
