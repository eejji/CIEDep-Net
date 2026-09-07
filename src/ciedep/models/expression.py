"""Expression stage (논문 III-D).

Conformer 블록으로 국소 음향 패턴과 장기 운율 의존성을 함께 모델링한 뒤,
adaptive average pooling + linear regression 으로 PHQ-8 점수를 예측한다.

Baseline 4 (Transformer 백본) 비교를 위해 TransformerExpressionAnalyzer 도 제공한다.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .conformer import ConformerBlock


class ConformerExpressionAnalyzer(nn.Module):
    """제안 모델의 표현 분석 블록 (Conformer 4 layers, 4 heads, kernel 31)."""

    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_expansion: int = 2,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=d_model,
                    num_attention_heads=num_heads,
                    feed_forward_expansion_factor=ff_expansion,
                    conv_expansion_factor=conv_expansion,
                    feed_forward_dropout_rate=dropout,
                    attention_dropout_rate=dropout,
                    conv_dropout_rate=dropout,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(d_model, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)          # (B, T, D) -> (B, D, T)
        x = self.pool(x).squeeze(-1)   # (B, D)
        return self.fc(x)


class TransformerExpressionAnalyzer(nn.Module):
    """Baseline 4: Conformer 대신 표준 Transformer 를 쓰는 표현 분석 블록."""

    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
