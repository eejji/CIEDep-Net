"""Whisper-large 로 턴 단위 오디오를 전사해 Q-A 대화 구조를 만든다 (논문 III-B-1, IV-B).

턴 단위로 분리한 오디오를 순차적으로 Whisper 에 넣고,
각 텍스트 앞에 화자 라벨을 붙여 대화 구조를 복원한다.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from ..data.turns import extract_turn_pairs, make_dialogue_transcript, slice_turn_audio


class WhisperTranscriber:
    """openai-whisper 래퍼."""

    def __init__(self, model_name: str = "large", device: Optional[str] = None) -> None:
        import torch
        import whisper

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe_array(self, audio, language: str = "en") -> str:
        """float32 mono 16kHz 배열을 전사한다."""
        import numpy as np

        if audio is None or len(audio) == 0:
            return ""
        result = self.model.transcribe(np.asarray(audio, dtype="float32"), language=language, fp16=False)
        return str(result.get("text", "")).strip()

    def transcribe_file(self, path: str | os.PathLike, language: str = "en") -> str:
        result = self.model.transcribe(str(path), language=language, fp16=False)
        return str(result.get("text", "")).strip()


def transcribe_participant(
    transcriber: WhisperTranscriber,
    waveform,
    transcript_df: pd.DataFrame,
    columns: Dict[str, str],
    interviewer_label: str = "Ellie",
    participant_label: str = "Participant",
    output_interviewer_label: str = "Interviewer",
    sample_rate: int = 16000,
    language: str = "en",
) -> str:
    """한 참가자의 턴 단위 전사를 만들어 Q-A 대화 문자열로 반환한다."""
    turns = extract_turn_pairs(
        transcript_df, columns, interviewer_label=interviewer_label, participant_label=participant_label
    )

    interviewer_texts: List[str] = []
    participant_texts: List[str] = []

    for turn in turns:
        interviewer_audio = slice_turn_audio(waveform, turn["interviewer_times"], sample_rate)
        participant_audio = slice_turn_audio(waveform, turn["participant_times"], sample_rate)
        interviewer_texts.append(transcriber.transcribe_array(interviewer_audio, language))
        participant_texts.append(transcriber.transcribe_array(participant_audio, language))

    return make_dialogue_transcript(
        interviewer_texts,
        participant_texts,
        interviewer_label=output_interviewer_label,
        participant_label=participant_label,
    )


def dialogue_from_reference_transcript(
    transcript_df: pd.DataFrame,
    columns: Dict[str, str],
    interviewer_label: str = "Ellie",
    participant_label: str = "Participant",
    output_interviewer_label: str = "Interviewer",
) -> str:
    """Whisper 없이 배포본 전사 텍스트만으로 Q-A 구조를 만든다 (빠른 확인용)."""
    turns = extract_turn_pairs(
        transcript_df, columns, interviewer_label=interviewer_label, participant_label=participant_label
    )
    interviewer_texts = [" ".join(str(t) for t in turn["interviewer_texts"]) for turn in turns]
    participant_texts = [" ".join(str(t) for t in turn["participant_texts"]) for turn in turns]
    return make_dialogue_transcript(
        interviewer_texts,
        participant_texts,
        interviewer_label=output_interviewer_label,
        participant_label=participant_label,
    )


def extract_participant_utterances(dialogue: str, participant_label: str = "Participant") -> List[str]:
    """Q-A 대화 문자열에서 참가자 발화만 뽑는다 (BERTScore 참조문 구성용)."""
    import re

    pattern = rf"{participant_label}:\s*(.*)"
    return [m.strip() for m in re.findall(pattern, dialogue) if m.strip()]
