"""CIEDep-Net 전체 모델 (논문 Fig. 1).

    Mel spectrogram ─► Cognition (Bi-LSTM)
                          │
        LLM depression score ─► SC-Fusion  (Eq. 5-7)
                          │
        LLM inner summary  ─► CA-Fusion  (Eq. 8-12)
                          │
                       Interpretation (Transformer)
                          │
        Wav2Vec 2.0 feature ─► CA-Fusion (Eq. 8-12)
                          │
                       Expression (Conformer) ─► PHQ-8

`ablation` 인자로 논문 Table III / IV 의 구성들을 재현한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .cognition import CognitionAnalyzer
from .expression import ConformerExpressionAnalyzer, TransformerExpressionAnalyzer
from .fusion import CrossAttentionFusion, ScoreConditionedFusion
from .interpretation import InterpretationAnalyzer


@dataclass
class AblationFlags:
    """논문 Table III / IV 의 구성 스위치.

    use_cognition      : Cognition stage(Bi-LSTM) 사용 여부. 끄면 표현 특징을
                         d_model 로 투영한 시퀀스를 주 경로로 사용한다.
    use_interpretation : Interpretation stage(LLM 특징 + Transformer) 사용 여부.
    use_expression     : Expression 특징(Wav2Vec 2.0)과의 CA-Fusion 사용 여부.
    use_score          : LLM 우울증 점수(SC-Fusion) 사용 여부.
    use_summary        : LLM 내면 요약(CA-Fusion) 사용 여부.
    fusion             : 'sc_ca'(제안) 또는 'concat'(Baseline 3).
    expression_backbone: 'conformer'(제안) 또는 'transformer'(Baseline 4).
    """

    use_cognition: bool = True
    use_interpretation: bool = True
    use_expression: bool = True
    use_score: bool = True
    use_summary: bool = True
    fusion: str = "sc_ca"
    expression_backbone: str = "conformer"


class ConcatFusion(nn.Module):
    """Baseline 3: SC/CA-Fusion 대신 단순 결합 후 선형 투영."""

    def __init__(self, d_model: int, d_aux: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model + d_aux, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, main_seq: Tensor, aux_feat: Tensor, return_attn: bool = False):
        B, T, _ = main_seq.shape
        if aux_feat.dim() == 2:
            aux_feat = aux_feat.unsqueeze(1).expand(-1, T, -1)
        out = self.proj(torch.cat([main_seq, aux_feat], dim=-1))
        if return_attn:
            return out, None, None
        return out


class CIEDepNet(nn.Module):
    """Cognition-Interpretation-Expression Depression Network."""

    def __init__(self, config: Dict[str, Any], ablation: Optional[AblationFlags] = None) -> None:
        super().__init__()
        self.cfg = config
        self.ablation = ablation or AblationFlags(
            fusion=config.get("fusion", "sc_ca"),
            expression_backbone=config.get("expression_backbone", "conformer"),
        )

        d_expr = config["d_expr"]
        d_summary = config["d_summary"]
        d_output = config.get("d_output", 1)
        dropout = config.get("dropout", 0.1)

        # ---------------- Cognition ----------------
        if self.ablation.use_cognition:
            self.cognition = CognitionAnalyzer(
                d_input=config["d_input"],
                hidden_dims=config["lstm_hidden_dims"],
                bidirectional=config.get("lstm_bidirectional", True),
            )
            d_model = self.cognition.output_dim
        else:
            # 인지 단계를 제거하면 표현 특징 시퀀스를 주 경로로 사용한다.
            d_model = config["lstm_hidden_dims"][-1] * (2 if config.get("lstm_bidirectional", True) else 1)
            self.cognition = None
            self.expr_entry_proj = nn.Linear(d_expr, d_model)

        self.d_model = d_model

        # ------------- Interpretation --------------
        use_sc_ca = self.ablation.fusion == "sc_ca"

        if self.ablation.use_interpretation:
            if self.ablation.use_score:
                self.score_fusion = (
                    ScoreConditionedFusion(d_cognition=d_model)
                    if use_sc_ca
                    else ConcatFusion(d_model, 1, dropout)
                )
            if self.ablation.use_summary:
                self.summary_fusion = (
                    CrossAttentionFusion(
                        d_model=d_model,
                        d_aux=d_summary,
                        n_heads=config.get("summary_transformer_heads", 4),
                        dropout=dropout,
                    )
                    if use_sc_ca
                    else ConcatFusion(d_model, d_summary, dropout)
                )

            self.interpretation = InterpretationAnalyzer(
                d_model=d_model,
                n_heads=config.get("transformer_heads", 4),
                dim_feedforward=config.get("transformer_ff_dim", 512),
                n_layers=config.get("transformer_layers", 2),
                dropout=dropout,
            )

        # --------------- Expression ----------------
        if self.ablation.use_expression and self.ablation.use_cognition:
            self.expr_fusion = (
                CrossAttentionFusion(d_model=d_model, d_aux=d_expr, n_heads=4, dropout=dropout)
                if use_sc_ca
                else ConcatFusion(d_model, d_expr, dropout)
            )

        if self.ablation.expression_backbone == "conformer":
            self.expression = ConformerExpressionAnalyzer(
                d_model=d_model,
                out_dim=d_output,
                num_layers=config.get("conformer_layers", 4),
                num_heads=config.get("conformer_heads", 4),
                ff_expansion=config.get("conformer_ff_expansion", 4),
                conv_expansion=config.get("conformer_conv_expansion", 2),
                dropout=dropout,
                conv_kernel_size=config.get("conformer_kernel_size", 31),
                half_step_residual=config.get("half_step_residual", True),
            )
        else:
            self.expression = TransformerExpressionAnalyzer(
                d_model=d_model,
                out_dim=d_output,
                num_layers=config.get("conformer_layers", 4),
                num_heads=config.get("conformer_heads", 4),
                dim_feedforward=config.get("transformer_ff_dim", 512),
                dropout=dropout,
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        mel: Tensor,
        score: Tensor,
        summary: Tensor,
        expr_feat: Tensor,
        return_attn: bool = False,
    ):
        """
        mel       : (B, T, d_input)   인지 단계 입력 (멜 스펙트로그램)
        score     : (B, 1)            LLM 우울증 점수
        summary   : (B, d_summary)    LLM 내면 요약 문장 임베딩
        expr_feat : (B, T, d_expr)    Wav2Vec 2.0 표현 특징
        """
        attn_main, attn_aux = None, None

        # Step 1: Cognition
        if self.cognition is not None:
            x = self.cognition(mel)
        else:
            x = self.expr_entry_proj(expr_feat)

        # Step 2-4: Interpretation (SC-Fusion -> CA-Fusion -> Transformer)
        if self.ablation.use_interpretation:
            if self.ablation.use_score:
                x = self.score_fusion(x, score)
            if self.ablation.use_summary:
                x = self.summary_fusion(x, summary)
            x = self.interpretation(x)

        # Step 5: Expression fusion
        if self.ablation.use_expression and self.cognition is not None:
            out = self.expr_fusion(x, expr_feat, return_attn=return_attn)
            if return_attn:
                x, attn_main, attn_aux = out
            else:
                x = out

        # Step 6: Expression analyzer -> PHQ-8
        prediction = self.expression(x)

        if return_attn:
            return prediction, attn_main, attn_aux
        return prediction

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, mel: Tensor, score: Tensor, summary: Tensor, expr_feat: Tensor) -> Dict[str, Tensor]:
        """t-SNE 시각화를 위한 단계별 중간 표현 추출."""
        feats: Dict[str, Tensor] = {"raw_input": mel.mean(dim=1).detach().cpu()}

        x = self.cognition(mel) if self.cognition is not None else self.expr_entry_proj(expr_feat)
        feats["cognition"] = x.mean(dim=1).detach().cpu()

        if self.ablation.use_interpretation:
            if self.ablation.use_score:
                x = self.score_fusion(x, score)
                feats["score_fused"] = x.mean(dim=1).detach().cpu()
            if self.ablation.use_summary:
                x = self.summary_fusion(x, summary)
                feats["summary_fused"] = x.mean(dim=1).detach().cpu()
            x = self.interpretation(x)
            feats["interpretation"] = x.mean(dim=1).detach().cpu()

        if self.ablation.use_expression and self.cognition is not None:
            x = self.expr_fusion(x, expr_feat)
            feats["final"] = x.mean(dim=1).detach().cpu()

        return feats


class TextOnlyModel(nn.Module):
    """Table III 의 'Interpretation only' (T 모달리티) 구성."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        d_summary = config["d_summary"]
        self.score_fusion = ScoreConditionedFusion(d_cognition=d_summary)
        self.head = nn.Sequential(
            nn.Linear(d_summary, d_summary // 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(d_summary // 2, config.get("d_output", 1)),
        )

    def forward(self, score: Tensor, summary: Tensor) -> Tensor:
        x = self.score_fusion(summary.unsqueeze(1), score).squeeze(1)
        return self.head(x)


def build_model(config: Dict[str, Any], ablation: Optional[AblationFlags] = None) -> nn.Module:
    """설정에서 모델을 만든다."""
    flags = ablation or AblationFlags(
        fusion=config.get("fusion", "sc_ca"),
        expression_backbone=config.get("expression_backbone", "conformer"),
    )
    if flags.use_interpretation and not (flags.use_cognition or flags.use_expression):
        return TextOnlyModel(config)
    return CIEDepNet(config, flags)
