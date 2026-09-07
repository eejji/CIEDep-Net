"""LLM 산출물의 예측 타당성 검증 (논문 Abstract, Table II).

- 우울증 점수 : 생성 점수 x 24 를 실제 PHQ-8 과 비교 (논문 r = 0.69, p < 0.01)
- 내면 요약   : 참가자 발화 전체를 참조문으로 BERTScore F1 계산 (논문 0.8)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..llm.parse import score_to_phq
from ..metrics import regression_metrics


def evaluate_generated_scores(
    generated: Sequence[float],
    actual_phq: Sequence[float],
    scale: float = 24.0,
) -> Dict[str, float]:
    """0~1 생성 점수를 PHQ-8 범위로 환산해 실제 점수와 비교한다."""
    predicted = np.array([score_to_phq(s, scale) for s in generated], dtype=float)
    return regression_metrics(np.asarray(actual_phq, dtype=float), predicted)


def bertscore_summaries(
    summaries: Sequence[str],
    references: Sequence[str],
    lang: str = "en",
    rescale_with_baseline: bool = False,
    batch_size: int = 16,
) -> Dict[str, object]:
    """내면 요약과 참가자 발화 사이의 BERTScore F1 을 계산한다.

    references[i] 는 참가자 i 의 발화를 모두 이어 붙인 문자열이다.
    """
    from bert_score import BERTScorer

    scorer = BERTScorer(lang=lang, rescale_with_baseline=rescale_with_baseline, batch_size=batch_size)
    _, _, f1 = scorer.score(list(summaries), list(references))
    f1_list: List[float] = [float(v) for v in f1]

    return {
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "f1": f1_list,
    }


def compare_prompt_strategies(
    strategy_scores: Dict[str, Sequence[float]],
    actual_phq: Sequence[float],
    scale: float = 24.0,
) -> Dict[str, Dict[str, float]]:
    """Table II 처럼 프롬프트 전략별 성능을 한 번에 비교한다."""
    return {
        name: evaluate_generated_scores(scores, actual_phq, scale)
        for name, scores in strategy_scores.items()
    }
