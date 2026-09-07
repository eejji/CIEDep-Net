"""Expression stage 입력: SSL 음성 표현 (논문 III-D-1, Table I).

Wav2Vec 2.0 / HuBERT / WavLM / UniSpeech-SAT 를 같은 인터페이스로 지원한다.
세그먼트별 last_hidden_state 의 프레임 평균을 취해 세그먼트당 벡터를 만든다.
논문 최종 모델은 4초 세그먼트 + Wav2Vec 2.0 (768차원)을 사용한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import torch

from .segment import segment_audio, window_sizes

DEFAULT_MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base-960h",
    "hubert": "facebook/hubert-large-ls960-ft",
    "wavlm": "microsoft/wavlm-large",
    "unispeech_sat": "microsoft/unispeech-sat-base",
}


class SSLFeatureExtractor:
    """transformers 기반 SSL 음성 인코더 래퍼."""

    def __init__(
        self,
        model_name: str = "wav2vec2",
        model_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = model_name
        self.model_id = model_id or DEFAULT_MODEL_IDS[model_name]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def encode_segments(self, segments: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """(n_segments, window) -> (n_segments, hidden_dim)."""
        vectors = []
        for segment in segments:
            inputs = self.processor(segment, sampling_rate=sample_rate, return_tensors="pt")
            input_values = inputs["input_values"].to(self.device)
            outputs = self.model(input_values)
            vectors.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy())
        return np.stack(vectors).astype(np.float32)

    def extract_file(
        self,
        audio_path: str | os.PathLike,
        sample_rate: int = 16000,
        window_sec: float = 4.0,
        hop_sec: float = 3.0,
    ) -> np.ndarray:
        waveform, _ = librosa.load(str(audio_path), sr=sample_rate)
        window_size, hop_size = window_sizes(sample_rate, window_sec, hop_sec)
        segments = segment_audio(waveform, window_size, hop_size)
        return self.encode_segments(segments, sample_rate)

    def extract_directories(
        self,
        audio_dirs,
        sample_rate: int = 16000,
        window_sec: float = 4.0,
        hop_sec: float = 3.0,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        features: Dict[str, np.ndarray] = {}
        for directory in audio_dirs:
            directory = Path(directory)
            if not directory.exists():
                print(f"[warn] 경로가 없습니다: {directory}")
                continue
            for path in sorted(directory.glob("*.wav")):
                features[path.stem] = self.extract_file(path, sample_rate, window_sec, hop_sec)
                if verbose:
                    print(f"[{self.model_name}] {path.stem}: {features[path.stem].shape}")
        return features
