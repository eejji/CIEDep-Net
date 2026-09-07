"""Cognition stage (논문 III-A, Eq. 1-4).

멜 스펙트로그램을 입력으로 3층 Bi-LSTM + LayerNorm 을 반복해
음소/음절 -> 단어 -> 문장 수준으로 확장되는 청각 인지 흐름을 모사한다.
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch import Tensor


class CognitionAnalyzer(nn.Module):
    """Bi-LSTM + LayerNorm 스택.

    입력  : (B, T, d_input)   멜 스펙트로그램 세그먼트 시퀀스
    출력  : (B, T, 2 * hidden_dims[-1])
    """

    def __init__(
        self,
        d_input: int = 80,
        hidden_dims: Sequence[int] = (64, 128, 256),
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        lstm_layers, norm_layers = [], []
        d_cur = d_input
        for d_hidden in hidden_dims:
            lstm_layers.append(
                nn.LSTM(
                    input_size=d_cur,
                    hidden_size=d_hidden,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            d_cur = d_hidden * (2 if bidirectional else 1)
            norm_layers.append(nn.LayerNorm(d_cur))

        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.output_dim = d_cur

    def forward(self, x: Tensor) -> Tensor:
        for lstm, norm in zip(self.lstm_layers, self.norm_layers):
            x, _ = lstm(x)
            x = norm(x)
        return x
