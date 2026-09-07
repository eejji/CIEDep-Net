"""Cognition stage 입력: 멜 스펙트로그램 (논문 III-A-1).

세그먼트마다 80-bin 멜 스펙트로그램(dB)을 구하고 시간축 평균을 취해
세그먼트당 80차원 벡터를 만든 뒤, 참가자 단위로 min-max 정규화한다.
결과: (n_segments, 80)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from .segment import segment_audio, window_sizes


def build_mel_transform(
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_sec: float = 0.025,
    frame_stride_sec: float = 0.010,
) -> T.MelSpectrogram:
    n_fft = int(round(sample_rate * frame_length_sec))
    hop_length = int(round(sample_rate * frame_stride_sec))
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )


def extract_mel_features(
    audio_path: str | os.PathLike,
    mel_transform: T.MelSpectrogram,
    sample_rate: int = 16000,
    window_sec: float = 4.0,
    hop_sec: float = 3.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """한 음성 파일에서 (n_segments, n_mels) 특징을 뽑는다."""
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.size(0) > 1:                       # 스테레오면 모노로
        waveform = waveform.mean(dim=0, keepdim=True)

    window_size, hop_size = window_sizes(sample_rate, window_sec, hop_sec)
    segments = segment_audio(waveform.squeeze(0).numpy(), window_size, hop_size)

    db_transform = T.AmplitudeToDB(stype="power", top_db=top_db)
    vectors = []
    for segment in segments:
        mel = mel_transform(torch.from_numpy(segment).float())
        mel_db = db_transform(mel)
        vectors.append(mel_db.mean(dim=-1))        # 시간축 평균 -> (n_mels,)

    features = torch.stack(vectors).numpy()

    # 참가자 단위 min-max 정규화
    lo, hi = features.min(), features.max()
    if hi - lo == 0:
        return np.zeros_like(features)
    return ((features - lo) / (hi - lo)).astype(np.float32)


def extract_mel_directory(
    audio_dirs,
    mel_transform: T.MelSpectrogram,
    sample_rate: int = 16000,
    window_sec: float = 4.0,
    hop_sec: float = 3.0,
    top_db: float = 80.0,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """여러 디렉터리의 wav 를 모두 처리해 {sample_id: (T, 80)} 사전을 만든다."""
    features: Dict[str, np.ndarray] = {}
    for directory in audio_dirs:
        directory = Path(directory)
        if not directory.exists():
            print(f"[warn] 경로가 없습니다: {directory}")
            continue
        for path in sorted(directory.glob("*.wav")):
            features[path.stem] = extract_mel_features(
                path, mel_transform, sample_rate, window_sec, hop_sec, top_db
            )
            if verbose:
                print(f"[mel] {path.stem}: {features[path.stem].shape}")
    return features
