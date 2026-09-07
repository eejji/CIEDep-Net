"""샘플 목록 구성, 특징 정렬, PyTorch Dataset, 학습/검증/평가 분할.

특징 파일은 {sample_id: np.ndarray} 형태의 pickle 로 저장한다.
sample_id 는 확장자를 뺀 파일명이며(예: '300_P', '312_P_pitch_shift_1'),
참가자 ID 는 sample_id 앞의 숫자에서 얻는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .labels import parse_participant_id, phq_score_to_class


@dataclass
class Sample:
    """한 학습 샘플 (원본 참가자 음성 또는 그 증강본)."""

    sample_id: str
    participant_id: str
    path: str
    phq_score: float
    phq_class: int
    is_augmented: bool


def build_sample_index(
    participant_dir: str | os.PathLike,
    augment_dirs: Sequence[str | os.PathLike],
    label_lookup: Dict[str, Dict[str, float]],
    class_bins: Sequence[float] = (9, 14),
    verbose: bool = True,
) -> List[Sample]:
    """참가자 음성 + 증강 음성을 훑어 라벨이 붙은 샘플 목록을 만든다.

    라벨 조회에 실패한 파일(제외 참가자 등)은 건너뛴다.
    """
    samples: List[Sample] = []
    skipped: List[str] = []

    for is_aug, directory in [(False, participant_dir)] + [(True, d) for d in augment_dirs]:
        directory = Path(directory)
        if not directory.exists():
            print(f"[warn] 경로가 없습니다: {directory}")
            continue

        for path in sorted(directory.glob("*.wav")):
            sample_id = path.stem
            try:
                pid = parse_participant_id(sample_id)
            except ValueError:
                skipped.append(sample_id)
                continue

            info = label_lookup.get(pid)
            if info is None:
                skipped.append(sample_id)
                continue

            samples.append(
                Sample(
                    sample_id=sample_id,
                    participant_id=pid,
                    path=str(path),
                    phq_score=float(info["phq_score"]),
                    phq_class=int(info.get("phq_class", phq_score_to_class(info["phq_score"], class_bins))),
                    is_augmented=is_aug,
                )
            )

    if verbose:
        n_aug = sum(s.is_augmented for s in samples)
        print(f"샘플 {len(samples)}개 (원본 {len(samples) - n_aug}, 증강 {n_aug}), 제외 {len(skipped)}개")
    return samples


# ---------------------------------------------------------------------------
# 시퀀스 길이 정규화
# ---------------------------------------------------------------------------
def pad_or_truncate(sequence: np.ndarray, length: int) -> np.ndarray:
    """(T, D) 시퀀스를 (length, D) 로 자르거나 0으로 채운다."""
    sequence = np.asarray(sequence, dtype=np.float32)
    T, D = sequence.shape
    if T >= length:
        return sequence[:length]
    return np.vstack([sequence, np.zeros((length - T, D), dtype=np.float32)])


def stack_features(
    feature_dict: Dict[str, np.ndarray],
    samples: Sequence[Sample],
    length: int,
) -> np.ndarray:
    """샘플 순서에 맞춰 특징을 정렬하고 길이를 맞춰 (N, length, D) 로 쌓는다."""
    missing = [s.sample_id for s in samples if s.sample_id not in feature_dict]
    if missing:
        raise KeyError(f"특징이 없는 샘플 {len(missing)}개 (예: {missing[:5]})")
    return np.stack([pad_or_truncate(feature_dict[s.sample_id], length) for s in samples])


def align_participant_level(
    value_dict: Dict[str, object],
    samples: Sequence[Sample],
) -> np.ndarray:
    """참가자 단위 값(LLM 점수, 요약 임베딩)을 샘플 순서로 펼친다.

    증강 샘플은 원본 참가자와 같은 값을 공유한다.
    """
    missing = sorted({s.participant_id for s in samples if s.participant_id not in value_dict})
    if missing:
        raise KeyError(f"LLM 산출물이 없는 참가자 {len(missing)}명 (예: {missing[:5]})")
    return np.array([value_dict[s.participant_id] for s in samples], dtype=np.float32)


def estimate_pad_length(feature_dict: Dict[str, np.ndarray]) -> int:
    """참가자당 세그먼트 수의 평균(논문: DAIC-WOZ 227, E-DAIC 159)."""
    return int(np.mean([np.asarray(v).shape[0] for v in feature_dict.values()]))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CIEDepDataset(Dataset):
    """CIEDep-Net 입력 묶음.

    mel     : (N, T, 80)   Cognition 입력
    score   : (N, 1)       LLM 우울증 점수
    summary : (N, 768)     LLM 내면 요약 문장 임베딩
    expr    : (N, T, 768)  Wav2Vec 2.0 표현 특징
    target  : (N, 1)       실제 PHQ-8 점수
    """

    def __init__(
        self,
        mel: np.ndarray,
        score: np.ndarray,
        summary: np.ndarray,
        expr: np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.mel = torch.as_tensor(np.asarray(mel), dtype=torch.float32)
        self.score = torch.as_tensor(np.asarray(score), dtype=torch.float32).reshape(-1, 1)
        self.summary = torch.as_tensor(np.asarray(summary), dtype=torch.float32)
        self.expr = torch.as_tensor(np.asarray(expr), dtype=torch.float32)
        self.target = torch.as_tensor(np.asarray(target), dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.mel)

    def __getitem__(self, idx: int):
        return self.mel[idx], self.score[idx], self.summary[idx], self.expr[idx], self.target[idx]


# ---------------------------------------------------------------------------
# 분할
# ---------------------------------------------------------------------------
def train_test_indices(
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 128,
) -> Tuple[List[int], List[int]]:
    """PHQ-8 3구간으로 층화한 train+val / test 분할 인덱스."""
    return train_test_split(
        list(range(len(labels))),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def stratified_folds(
    train_val_indices: Sequence[int],
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 128,
):
    """train+val 구간에 대한 StratifiedKFold 제너레이터."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_labels = labels[list(train_val_indices)]
    for fold, (train_idx, val_idx) in enumerate(skf.split(list(train_val_indices), fold_labels)):
        yield (
            fold,
            [train_val_indices[i] for i in train_idx],
            [train_val_indices[i] for i in val_idx],
        )


def scale_expression_features(
    train: np.ndarray,
    others: Sequence[np.ndarray] = (),
) -> Tuple[np.ndarray, List[np.ndarray], StandardScaler]:
    """fold 단위 StandardScaler 를 학습 데이터에만 적합시키고 나머지에 적용한다."""
    scaler = StandardScaler()
    D = train.shape[-1]
    scaler.fit(train.reshape(-1, D))

    scaled_train = scaler.transform(train.reshape(-1, D)).reshape(train.shape).astype(np.float32)
    scaled_others = [scaler.transform(a.reshape(-1, D)).reshape(a.shape).astype(np.float32) for a in others]
    return scaled_train, scaled_others, scaler


def make_loader(dataset: Dataset, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
