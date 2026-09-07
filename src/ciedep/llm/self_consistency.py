"""Self-consistency 전략 (논문 III-B-2).

같은 프롬프트로 5개의 출력을 생성한 뒤,
  - 우울증 점수 : 5개 값의 평균을 최종 점수로 쓴다.
  - 내면 요약   : 5개 임베딩의 평균 벡터에 가장 가까운 출력을 대표로 고른다.
생성 다양성을 위해 temperature=0.8, top_p=0.9, max_new_tokens=1024 를 사용한다.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .parse import parse_depression_score, parse_inner_summary


def aggregate_scores(scores: Sequence[Optional[float]]) -> Optional[float]:
    """파싱에 성공한 점수들의 평균. 전부 실패하면 None."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def select_representative_summary(
    summaries: Sequence[str],
    embed_fn: Callable[[Sequence[str]], np.ndarray],
) -> Tuple[str, int]:
    """평균 임베딩에 가장 가까운(코사인 유사도 최대) 요약을 고른다."""
    texts = [s for s in summaries if s and s.strip()]
    if not texts:
        return "", -1
    if len(texts) == 1:
        return texts[0], 0

    embeddings = np.asarray(embed_fn(texts), dtype=np.float64)
    centroid = embeddings.mean(axis=0)

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid)
    norms[norms == 0] = 1e-12
    similarities = embeddings @ centroid / norms

    best = int(np.argmax(similarities))
    return texts[best], best


def run_self_consistency_score(
    runner,
    messages,
    n_samples: int = 5,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 32,
) -> Tuple[Optional[float], List[Optional[float]]]:
    """점수 self-consistency: n_samples 개 생성 후 평균."""
    outputs = runner.generate(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
    )
    scores = [parse_depression_score(text) for text in outputs]
    return aggregate_scores(scores), scores


def run_self_consistency_summary(
    runner,
    messages,
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    n_samples: int = 5,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
) -> Tuple[str, List[str]]:
    """요약 self-consistency: n_samples 개 생성 후 대표 1개 선택."""
    outputs = runner.generate(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
    )
    summaries = [parse_inner_summary(text) for text in outputs]
    representative, _ = select_representative_summary(summaries, embed_fn)
    return representative, summaries
