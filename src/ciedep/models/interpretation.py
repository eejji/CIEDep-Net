"""Interpretation stage (논문 III-B-3).

SC-Fusion / CA-Fusion 으로 결합된 인지-해석 표현을 Transformer encoder 로 처리해
단계 간 상호작용을 모델링한다. (2 layers, 4 heads, FFN 512)
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class InterpretationAnalyzer(nn.Module):
    """토큰 수준 특징을 유지한 채 전역 문맥 의존성을 학습한다.

    입력  : (B, T, d_model)
    출력  : (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer(x)
