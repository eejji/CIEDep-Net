"""Conformer 구성 요소 (논문 Eq. 13-16, Expression stage 백본).

Macaron-style FFN(half-step residual) - MHSA(상대 위치 인코딩) - Convolution - FFN - LayerNorm.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


class Swish(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Linear(nn.Module):
    """Xavier 초기화가 적용된 Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class Transpose(nn.Module):
    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class ResidualConnectionModule(nn.Module):
    """module(x) * module_factor + x * input_factor."""

    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0) -> None:
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class FeedForwardModule(nn.Module):
    """(batch, time, dim) -> (batch, time, dim)."""

    def __init__(self, encoder_dim: int = 512, expansion_factor: int = 4, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class RelPositionalEncoding(nn.Module):
    """상대 위치 인코딩 (Transformer-XL 방식)."""

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.pe: Optional[Tensor] = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(100000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1).to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        self.extend_pe(x)
        center = self.pe.size(1) // 2
        return self.pe[:, center - x.size(1) + 1 : center + x.size(1)]


class RelativeMultiHeadAttention(nn.Module):
    """상대 위치 편향(u_bias, v_bias)을 갖는 멀티헤드 어텐션."""

    def __init__(self, d_model: int = 512, num_heads: int = 4, dropout_rate: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        init.xavier_uniform_(self.u_bias)
        init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.unsqueeze(1), -1e9)

        attn = self.dropout(F.softmax(score, dim=-1))
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(context)

    @staticmethod
    def _relative_shift(pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded = torch.cat([zeros, pos_score], dim=-1)
        padded = padded.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        return padded[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs).repeat(batch_size, 1, 1)
        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)


class PointwiseConv1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    """LayerNorm - Pointwise - GLU - Depthwise - Swish - Pointwise - Dropout."""

    def __init__(
        self, in_channels: int, kernel_size: int = 31, expansion_factor: int = 2, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd for SAME padding"
        assert expansion_factor == 2, "currently only expansion_factor 2 is supported"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)


class ConformerBlock(nn.Module):
    """논문 Eq. (13)-(16) 의 conformer block."""

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 4,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        conv_dropout_rate: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ) -> None:
        super().__init__()
        ff_residual_factor = 0.5 if half_step_residual else 1.0

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_rate),
                module_factor=ff_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(encoder_dim, num_attention_heads, attention_dropout_rate)
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(encoder_dim, conv_kernel_size, conv_expansion_factor, conv_dropout_rate)
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_rate),
                module_factor=ff_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
