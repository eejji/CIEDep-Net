"""PHQ-8 라벨 로딩과 3구간 클래스 변환.

논문 IV-B: 0-9 non-depressed, 10-14 moderate, 15-24 severe.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def phq_score_to_class(score: float, bins: Sequence[float] = (9, 14)) -> int:
    """PHQ-8 점수를 3구간 클래스(0/1/2)로 변환한다."""
    if score <= bins[0]:
        return 0
    if score <= bins[1]:
        return 1
    return 2


def load_labels(csv_path: str, columns: Dict[str, str], excluded_indices: Sequence[int] = ()) -> pd.DataFrame:
    """라벨 CSV를 읽어 participant_id / phq_score / phq_class 열을 가진 DataFrame 을 만든다.

    excluded_indices 는 논문에서 제외한 참가자의 **행 인덱스**다
    (DAIC-WOZ: transcript 결손 3명 -> 189명에서 186명).
    """
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            columns["participant_id"]: "participant_id",
            columns["phq_score"]: "phq_score",
            columns["phq_class"]: "phq_class",
        }
    )
    df = df[["participant_id", "phq_score", "phq_class"]].dropna(subset=["participant_id"])
    df["participant_id"] = df["participant_id"].astype(int).astype(str)

    if len(excluded_indices):
        df = df.drop(index=list(excluded_indices), errors="ignore")

    return df.reset_index(drop=True)


def build_label_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """participant_id -> {'phq_score', 'phq_class'} 사전을 만든다."""
    return {
        str(row.participant_id): {"phq_score": float(row.phq_score), "phq_class": int(row.phq_class)}
        for row in df.itertuples()
    }


def scores_to_classes(scores: Sequence[float], bins: Sequence[float] = (9, 14)) -> np.ndarray:
    """StratifiedKFold 층화에 쓸 클래스 배열을 만든다."""
    return np.array([phq_score_to_class(s, bins) for s in scores], dtype=int)


def parse_participant_id(filename: str) -> str:
    """파일명에서 참가자 ID(앞 3자리 숫자)를 뽑는다.

    '300_P.wav', '300_P_pitch_shift_1.wav', '300' 모두 '300' 으로 정규화된다.
    """
    stem = filename.split("/")[-1].split("\\")[-1]
    digits: List[str] = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        raise ValueError(f"참가자 ID를 찾을 수 없습니다: {filename}")
    return "".join(digits)
