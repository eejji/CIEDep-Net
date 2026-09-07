"""LLM 출력 파싱: 우울증 점수와 내면 요약 추출."""

from __future__ import annotations

import re
from typing import Optional

SCORE_PATTERN = re.compile(r">>>?\s*Depression score:\s*([01](?:\.\d+)?)", re.IGNORECASE)
FALLBACK_SCORE_PATTERN = re.compile(r"\b([01]\.\d{1,3})\b")
SUMMARY_PATTERN = re.compile(r">>>?\s*Inner Monologue:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_depression_score(text: str) -> Optional[float]:
    """'>>> Depression score: 0.XX' 형식에서 0~1 점수를 뽑는다.

    형식이 어긋나면 본문에서 0~1 사이 소수를 한 번 더 찾아보고,
    그래도 없으면 None 을 반환한다(재시도 대상).
    """
    if not text:
        return None

    match = SCORE_PATTERN.search(text)
    if match is None:
        match = FALLBACK_SCORE_PATTERN.search(text)
    if match is None:
        return None

    score = float(match.group(1))
    return score if 0.0 <= score <= 1.0 else None


def parse_inner_summary(text: str) -> str:
    """'>>> Inner Monologue: ...' 뒤의 본문을 뽑고 태그를 정리한다."""
    if not text:
        return ""

    match = SUMMARY_PATTERN.search(text)
    summary = match.group(1) if match else text

    summary = re.sub(r"(?i)^\s*assistant\s*\n+", "", summary)
    summary = re.sub(r"(?i)\s*Inner Monologue:\s*", " ", summary)
    return summary.strip()


def score_to_phq(score: float, scale: float = 24.0) -> float:
    """0~1 생성 점수를 PHQ-8 (0~24) 범위로 환산한다."""
    return float(score) * scale
