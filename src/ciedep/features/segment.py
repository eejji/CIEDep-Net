"""슬라이딩 윈도우 세그먼테이션 (논문 IV-B).

참가자 전용 음성을 4초 윈도우 / 1초 중첩(= 3초 hop)으로 자른다.
Table I 에서 4초 세그먼트가 최적이었다.
"""

from __future__ import annotations

from typing import List

import numpy as np


def segment_audio(
    waveform: np.ndarray,
    window_size: int,
    hop_size: int,
) -> np.ndarray:
    """1D 파형을 (n_segments, window_size) 로 자른다.

    윈도우가 한 번도 들어가지 않을 만큼 짧은 음성은 0으로 채운 한 개의 세그먼트를 돌려준다.
    """
    waveform = np.asarray(waveform).reshape(-1)
    total = waveform.shape[0]

    if total < window_size:
        padded = np.zeros(window_size, dtype=waveform.dtype)
        padded[:total] = waveform
        return padded[None, :]

    segments: List[np.ndarray] = [
        waveform[start : start + window_size] for start in range(0, total - window_size + 1, hop_size)
    ]
    return np.stack(segments)


def window_sizes(sample_rate: int, window_sec: float, hop_sec: float) -> tuple[int, int]:
    """초 단위 설정을 샘플 수로 바꾼다."""
    return int(sample_rate * window_sec), int(sample_rate * hop_sec)
