"""Interpretation stage 입력: 내면 요약 문장 임베딩 (논문 III-B-3).

내면 요약에는 부정 어휘가 많아 단어 단위 분석이 의미를 왜곡할 수 있으므로
문장 수준 임베딩을 쓴다. 기본 모델은 all-mpnet-base-v2 (768차원).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# LLM 출력에 섞여 나오는 태그/접두어
_TAG_PATTERNS = [
    r">>>?\s*Inner Monologue:\s*",
    r"\s*Inner Monologue:\s*",
    r"(?i)^\s*assistant\s*\n+",
]


def clean_summary(text: str) -> str:
    """LLM 출력에서 형식 태그를 제거하고 앞뒤 공백을 정리한다."""
    cleaned = str(text)
    for pattern in _TAG_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def embed_summaries(
    summaries: Sequence[str],
    model_name: str = "all-mpnet-base-v2",
    device: Optional[torch.device] = None,
    clean: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """요약 문자열 목록을 (N, dim) 임베딩 행렬로 변환한다."""
    from sentence_transformers import SentenceTransformer

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name).to(device)

    texts: List[str] = [clean_summary(s) if clean else str(s) for s in summaries]
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    return np.asarray(embeddings, dtype=np.float32)


def embed_summary_dict(
    summary_dict: Dict[str, str],
    model_name: str = "all-mpnet-base-v2",
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """{participant_id: summary} -> {participant_id: embedding}."""
    keys = list(summary_dict.keys())
    matrix = embed_summaries([summary_dict[k] for k in keys], model_name=model_name, device=device)
    return {k: matrix[i] for i, k in enumerate(keys)}
