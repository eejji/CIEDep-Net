"""화자 턴 분리와 Q-A 대화 구조 생성 (논문 IV-B, 텍스트 전처리 파이프라인).

1) 화자 턴 정의 : 한 화자의 발화 시작부터 상대 화자의 발화 시작 직전까지를 한 턴으로 본다.
2) 발화 시간 병합 : 한 턴 안의 여러 발화 구간을 원본 음성에서 잘라 이어 붙여 턴 오디오를 만든다.
3) Q-A 구조 생성 : 두 화자 모두에 적용해 완전한 질의-응답 구조를 만든다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import soundfile as sf


def extract_turn_pairs(
    transcript: pd.DataFrame,
    columns: Dict[str, str],
    interviewer_label: str = "Ellie",
    participant_label: str = "Participant",
) -> List[Dict[str, list]]:
    """전사 DataFrame 을 (면담자 턴, 참가자 턴) 쌍의 리스트로 변환한다."""
    speaker_col = columns["speaker"]
    text_col = columns["text"]
    start_col, stop_col = columns["start"], columns["stop"]

    df = transcript.reset_index(drop=True)
    turns: List[Dict[str, list]] = []
    i = 0

    while i < len(df):
        if str(df.loc[i, speaker_col]).strip() != interviewer_label:
            i += 1
            continue

        # 1) 면담자의 연속 발화 수집
        interviewer_texts = [df.loc[i, text_col]]
        interviewer_times = [(df.loc[i, start_col], df.loc[i, stop_col])]
        j = i + 1
        while j < len(df) and str(df.loc[j, speaker_col]).strip() == interviewer_label:
            interviewer_texts.append(df.loc[j, text_col])
            interviewer_times.append((df.loc[j, start_col], df.loc[j, stop_col]))
            j += 1

        # 2) 참가자의 연속 발화 수집
        participant_texts, participant_times = [], []
        k = j
        while k < len(df) and str(df.loc[k, speaker_col]).strip() == participant_label:
            participant_texts.append(df.loc[k, text_col])
            participant_times.append((df.loc[k, start_col], df.loc[k, stop_col]))
            k += 1

        # 3) 쌍이 형성될 때만 하나의 턴으로 인정
        if participant_texts:
            turns.append(
                {
                    "interviewer_texts": interviewer_texts,
                    "interviewer_times": interviewer_times,
                    "participant_texts": participant_texts,
                    "participant_times": participant_times,
                }
            )

        i = k if k > i else i + 1

    return turns


def slice_turn_audio(
    waveform: np.ndarray,
    times: Sequence[tuple],
    sample_rate: int = 16000,
) -> np.ndarray:
    """한 턴에 속한 발화 구간들을 원본 음성에서 잘라 이어 붙인다."""
    chunks = []
    for start, stop in times:
        s, e = int(float(start) * sample_rate), int(float(stop) * sample_rate)
        if e > s:
            chunks.append(waveform[s:e])
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def save_turn_audio(
    waveform: np.ndarray,
    turns: List[Dict[str, list]],
    output_dir: str | os.PathLike,
    participant_id: str,
    sample_rate: int = 16000,
) -> Dict[str, List[str]]:
    """턴별 오디오를 <output_dir>/<participant_id>/{interviewer,participant}/turn_N.wav 로 저장한다."""
    base = Path(output_dir) / participant_id
    paths: Dict[str, List[str]] = {"interviewer": [], "participant": []}

    for role, time_key in (("interviewer", "interviewer_times"), ("participant", "participant_times")):
        role_dir = base / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for idx, turn in enumerate(turns, start=1):
            audio = slice_turn_audio(waveform, turn[time_key], sample_rate)
            if audio.size == 0:
                continue
            out_path = role_dir / f"turn_{idx}.wav"
            sf.write(out_path, audio, sample_rate, format="wav")
            paths[role].append(str(out_path))

    return paths


def make_dialogue_transcript(
    interviewer_texts: Sequence[str],
    participant_texts: Sequence[str],
    interviewer_label: str = "Interviewer",
    participant_label: str = "Participant",
) -> str:
    """턴 단위 텍스트를 LLM 입력용 Q-A 대화 문자열로 조립한다.

    각 줄 앞에 화자 라벨을 붙여 LLM이 화자 역할을 명시적으로 인식하게 한다.
    """
    lines: List[str] = []
    n = min(len(interviewer_texts), len(participant_texts))

    for i in range(n):
        if str(interviewer_texts[i]).strip():
            lines.append(f"{interviewer_label}: {str(interviewer_texts[i]).strip()}")
        if str(participant_texts[i]).strip():
            lines.append(f"{participant_label}: {str(participant_texts[i]).strip()}")

    for i in range(n, len(interviewer_texts)):
        if str(interviewer_texts[i]).strip():
            lines.append(f"{interviewer_label}: {str(interviewer_texts[i]).strip()}")
    for i in range(n, len(participant_texts)):
        if str(participant_texts[i]).strip():
            lines.append(f"{participant_label}: {str(participant_texts[i]).strip()}")

    return "\n".join(lines)
