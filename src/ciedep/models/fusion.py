"""단계 간 융합 모듈: SC-Fusion (Eq. 5-7) 과 CA-Fusion (Eq. 8-12)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ScoreConditionedFusion(nn.Module):
    """SC-Fusion (논문 Fig. 3, Eq. 5-7).

    LLM이 생성한 우울증 점수 F_score (B, 1) 를 조건으로 인지 특징 F_cog (B, T, D) 를 변조한다.

        gamma = sigmoid(F_score W_gamma + b_gamma)
        beta  = F_score W_beta + b_beta
        F_CI1 = gamma * F_cog + beta
    """

    def __init__(self, d_cognition: int, d_score: int = 1) -> None:
        super().__init__()
        self.gamma_layer = nn.Sequential(nn.Linear(d_score, d_cognition), nn.Sigmoid())
        self.beta_layer = nn.Linear(d_score, d_cognition)

    def forward(self, cognition_seq: Tensor, score: Tensor) -> Tensor:
        B, T, D = cognition_seq.size()
        gamma = self.gamma_layer(score).unsqueeze(1).expand(B, T, D)
        beta = self.beta_layer(score).unsqueeze(1).expand(B, T, D)
        return gamma * cognition_seq + beta


class CrossAttentionFusion(nn.Module):
    """CA-Fusion (논문 Fig. 4, Eq. 8-12).

    주 특징 F1 (B, T, D) 과 보조 특징 F2 를 양방향 cross-attention 으로 융합한다.
    F2 는 (B, D2) 형태의 벡터(내면 요약 임베딩)이거나 (B, T, D2) 형태의 시퀀스
    (Wav2Vec 2.0 표현 특징)일 수 있으며, 두 경우 모두 D 차원으로 투영된다.

        A_{F1->F2} = MHSA(Q=F1, K=F2, V=F2)
        A_{F2->F1} = MHSA(Q=F2, K=F1, V=F1)
        Z = Dropout(GELU([A_{F1->F2}; A_{F2->F1}] W_f + b_f))
        X = LayerNorm(F1 + Z)
    """

    def __init__(self, d_model: int, d_aux: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.aux_proj = nn.Linear(d_aux, d_model)

        self.attn_main_to_aux = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_aux_to_main = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(
        self, main_seq: Tensor, aux_feat: Tensor, return_attn: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        B, T, D = main_seq.shape

        aux = self.aux_proj(aux_feat)
        if aux.dim() == 2:                     # (B, D) -> (B, T, D)
            aux = aux.unsqueeze(1).expand(-1, T, -1)

        main_to_aux, attn_m2a = self.attn_main_to_aux(
            query=main_seq, key=aux, value=aux, need_weights=return_attn
        )
        aux_to_main, attn_a2m = self.attn_aux_to_main(
            query=aux, key=main_seq, value=main_seq, need_weights=return_attn
        )

        fused = self.fusion_gate(torch.cat([main_to_aux, aux_to_main], dim=-1))
        out = self.fusion_norm(main_seq + fused)

        if return_attn:
            return out, attn_m2a, attn_a2m
        return out
