"""소수 클래스 오디오 증강 (논문 IV-B).

PHQ-8 3구간 중 moderate(10-14), severe(15-24) 에만 증강을 적용해 클래스 불균형을 완화한다.
- pitch shift  : ±3 semitone
- time stretch : 0.85 ~ 1.15 배
- time shift 는 시계열 순서를 왜곡하므로 사용하지 않는다.

DAIC-WOZ : moderate +99, severe +102  -> 총 387 샘플
E-DAIC   : moderate/severe 합계 +146  -> 총 567 샘플
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import librosa
import numpy as np
import soundfile as sf


def sample_pitch_shift(low: float = -3.0, high: float = 3.0, min_gap: float = 0.5) -> float:
    """0 근처(무변화)를 제외한 semitone 값을 뽑는다."""
    while True:
        value = random.uniform(low, high)
        if abs(value) >= min_gap:
            return value


def sample_time_stretch(low: float = 0.85, high: float = 1.15, min_gap: float = 0.05) -> float:
    """1 근처(무변화)를 제외한 배속 값을 뽑는다."""
    while True:
        value = random.uniform(low, high)
        if abs(value - 1.0) >= min_gap:
            return value


def apply_pitch_shift(audio: np.ndarray, sample_rate: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)


def apply_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(audio, rate=rate)


def augment_class(
    participant_ids: Sequence[str],
    source_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    n_extra: int,
    sample_rate: int = 16000,
    pitch_range: Sequence[float] = (-3.0, 3.0),
    pitch_min_gap: float = 0.5,
    stretch_range: Sequence[float] = (0.85, 1.15),
    stretch_min_gap: float = 0.05,
    seed: int = 128,
    verbose: bool = True,
) -> List[str]:
    """한 클래스에 대해 정확히 n_extra 개의 증강 샘플을 만든다.

    먼저 모든 참가자에 pitch shift 와 time stretch 를 한 번씩 적용하고
    (참가자 수 * 2 개), 목표치에 모자라면 무작위 참가자를 골라 채운다.
    반대로 목표치를 넘으면 앞에서부터 n_extra 개만 저장한다.
    """
    random.seed(seed)
    np.random.seed(seed)

    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load(pid: str) -> np.ndarray:
        for candidate in (f"{pid}.wav", f"{pid}_P.wav", f"{pid}_P", pid):
            path = source_dir / candidate
            if path.exists():
                audio, _ = librosa.load(path, sr=sample_rate)
                return audio
        raise FileNotFoundError(f"참가자 음성을 찾을 수 없습니다: {pid} ({source_dir})")

    # 1) 전 참가자에 두 기법을 한 번씩
    plan: List[tuple] = []
    for pid in participant_ids:
        plan.append((pid, "pitch_shift", 1))
        plan.append((pid, "time_stretch", 1))

    # 2) 부족분은 무작위 참가자로 채운다
    round_idx = 2
    while len(plan) < n_extra:
        remaining = n_extra - len(plan)
        picked = np.random.choice(list(participant_ids), size=min(remaining, len(participant_ids)), replace=False)
        for pid in picked:
            plan.append((str(pid), random.choice(["pitch_shift", "time_stretch"]), round_idx))
        round_idx += 1

    plan = plan[:n_extra]

    saved: List[str] = []
    for pid, method, idx in plan:
        audio = _load(pid)
        if method == "pitch_shift":
            value = sample_pitch_shift(pitch_range[0], pitch_range[1], pitch_min_gap)
            augmented = apply_pitch_shift(audio, sample_rate, value)
        else:
            value = sample_time_stretch(stretch_range[0], stretch_range[1], stretch_min_gap)
            augmented = apply_time_stretch(audio, value)

        out_path = output_dir / f"{pid}_{method}_{idx}.wav"
        sf.write(out_path, augmented, sample_rate, format="wav")
        saved.append(str(out_path))
        if verbose:
            print(f"[aug] {out_path.name}  ({method}={value:.3f})")

    return saved


def augment_minority_classes(
    label_lookup: Dict[str, Dict[str, float]],
    source_dir: str | os.PathLike,
    moderate_dir: str | os.PathLike,
    severe_dir: str | os.PathLike,
    moderate_extra: int,
    severe_extra: int,
    sample_rate: int = 16000,
    seed: int = 128,
    **augment_kwargs,
) -> Dict[str, List[str]]:
    """moderate / severe 클래스에 대해 증강을 수행한다."""
    moderate_ids = [pid for pid, info in label_lookup.items() if info["phq_class"] == 1]
    severe_ids = [pid for pid, info in label_lookup.items() if info["phq_class"] == 2]

    print(f"moderate {len(moderate_ids)}명 -> +{moderate_extra}, severe {len(severe_ids)}명 -> +{severe_extra}")

    return {
        "moderate": augment_class(
            moderate_ids, source_dir, moderate_dir, moderate_extra, sample_rate, seed=seed, **augment_kwargs
        ),
        "severe": augment_class(
            severe_ids, source_dir, severe_dir, severe_extra, sample_rate, seed=seed + 1, **augment_kwargs
        ),
    }
