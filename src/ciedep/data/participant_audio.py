"""전사(transcript) 시간 정보로 참가자 발화만 추출해 하나의 음성으로 잇는다.

논문 IV-B: "Only the participant's audio was extracted and all interviewer
speeches were removed to avoid potential bias in the depression analysis."

DAIC-WOZ  : transcript 에 speaker 열이 있어 Participant 행만 사용한다.
E-DAIC    : transcript 가 참가자 발화 구간만 담고 있어 모든 행을 사용한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def _find_pair(participant_dir: Path, audio_suffix: str, transcript_suffix: str):
    """참가자 폴더에서 (오디오, 전사) 파일 경로를 찾는다."""
    audio_path = transcript_path = None
    for entry in sorted(participant_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(audio_suffix) or (audio_path is None and name.lower().endswith(".wav")):
            audio_path = entry
        elif name.endswith(transcript_suffix) or (transcript_path is None and name.lower().endswith(".csv")):
            transcript_path = entry
    if audio_path is None or transcript_path is None:
        raise FileNotFoundError(f"오디오 또는 전사 파일을 찾을 수 없습니다: {participant_dir}")
    return audio_path, transcript_path


def extract_participant_audio(
    participant_dir: str | os.PathLike,
    columns: Dict[str, Optional[str]],
    participant_label: str = "Participant",
    sample_rate: int = 16000,
    audio_suffix: str = "_AUDIO.wav",
    transcript_suffix: str = "_TRANSCRIPT.csv",
    drop_last_row: bool = True,
) -> np.ndarray:
    """한 참가자의 발화 구간만 이어 붙인 파형을 반환한다."""
    participant_dir = Path(participant_dir)
    audio_path, transcript_path = _find_pair(participant_dir, audio_suffix, transcript_suffix)

    waveform, _ = librosa.load(audio_path, sr=sample_rate)
    df = pd.read_csv(transcript_path)

    speaker_col = columns.get("speaker")
    if speaker_col and speaker_col in df.columns:
        df = df[df[speaker_col].astype(str).str.strip() == participant_label]

    if drop_last_row and len(df) > 1:
        df = df.iloc[:-1]

    start_col, stop_col = columns["start"], columns["stop"]
    chunks: List[np.ndarray] = []
    for start, stop in zip(df[start_col], df[stop_col]):
        s, e = int(float(start) * sample_rate), int(float(stop) * sample_rate)
        if e > s:
            chunks.append(waveform[s:e])

    if not chunks:
        raise ValueError(f"참가자 발화 구간이 비어 있습니다: {participant_dir}")

    return np.concatenate(chunks)


def build_participant_dataset(
    raw_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    columns: Dict[str, Optional[str]],
    participant_label: str = "Participant",
    sample_rate: int = 16000,
    audio_suffix: str = "_AUDIO.wav",
    transcript_suffix: str = "_TRANSCRIPT.csv",
    verbose: bool = True,
) -> List[str]:
    """모든 참가자 폴더를 순회하며 참가자 전용 음성을 저장한다.

    반환값: 저장된 파일 경로 목록.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for participant_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        try:
            audio = extract_participant_audio(
                participant_dir,
                columns,
                participant_label=participant_label,
                sample_rate=sample_rate,
                audio_suffix=audio_suffix,
                transcript_suffix=transcript_suffix,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {participant_dir.name}: {exc}")
            continue

        out_path = output_dir / f"{participant_dir.name}.wav"
        sf.write(out_path, audio, sample_rate, format="wav")
        saved.append(str(out_path))
        if verbose:
            print(f"[ok] {participant_dir.name} -> {audio.shape[0] / sample_rate:.1f}s")

    return saved
