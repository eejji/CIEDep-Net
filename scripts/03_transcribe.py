"""3단계 - 턴 분리 + Whisper 전사 -> Q-A 대화 구조 생성.

전사의 시간 정보로 화자 턴을 정의하고, 턴 단위 오디오를 Whisper-large 로 전사한 뒤
"Interviewer:" / "Participant:" 라벨을 붙여 대화 구조를 만든다 (논문 IV-B).
결과는 {participant_id: dialogue} 형태의 pickle 로 저장된다.

    python scripts/03_transcribe.py --config configs/daic_woz.yaml
    python scripts/03_transcribe.py --config configs/daic_woz.yaml --use-reference-text
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import pandas as pd

import _bootstrap  # noqa: F401

from ciedep.config import load_config
from ciedep.llm import WhisperTranscriber, dialogue_from_reference_transcript, transcribe_participant
from ciedep.utils import save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="턴 단위 Whisper 전사")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--use-reference-text",
        action="store_true",
        help="Whisper 없이 배포본 전사 텍스트로 Q-A 구조만 만든다 (빠른 확인용)",
    )
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N명만 처리")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg, paths, audio_cfg = cfg["dataset"], cfg["paths"], cfg["audio"]
    columns = dataset_cfg["transcript_columns"]

    if columns.get("speaker") is None:
        raise SystemExit(
            "이 데이터셋의 전사에는 speaker 열이 없어 턴 분리를 할 수 없습니다.\n"
            "E-DAIC 는 배포본에 포함된 전사를 그대로 사용하세요."
        )

    transcriber = None if args.use_reference_text else WhisperTranscriber(cfg["features"]["whisper_model"])

    raw_dir = Path(dataset_cfg["raw_dir"])
    participant_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if args.limit:
        participant_dirs = participant_dirs[: args.limit]

    dialogues = {}
    for participant_dir in participant_dirs:
        pid = participant_dir.name.split("_")[0]
        audio_files = list(participant_dir.glob(f"*{dataset_cfg['audio_suffix']}")) or list(
            participant_dir.glob("*.wav")
        )
        transcript_files = list(participant_dir.glob(f"*{dataset_cfg['transcript_suffix']}")) or list(
            participant_dir.glob("*.csv")
        )
        if not audio_files or not transcript_files:
            print(f"[skip] {participant_dir.name}: 오디오/전사 파일 없음")
            continue

        transcript_df = pd.read_csv(transcript_files[0])

        if args.use_reference_text:
            dialogue = dialogue_from_reference_transcript(
                transcript_df,
                columns,
                interviewer_label=dataset_cfg["interviewer_label"],
                participant_label=dataset_cfg["participant_label"],
            )
        else:
            waveform, _ = librosa.load(audio_files[0], sr=audio_cfg["sample_rate"])
            dialogue = transcribe_participant(
                transcriber,
                waveform,
                transcript_df,
                columns,
                interviewer_label=dataset_cfg["interviewer_label"],
                participant_label=dataset_cfg["participant_label"],
                sample_rate=audio_cfg["sample_rate"],
            )

        dialogues[pid] = dialogue
        print(f"[ok] {pid}: {len(dialogue.splitlines())} lines")

    save_pickle(dialogues, paths["transcript_pkl"])
    print(f"\n전사 {len(dialogues)}건 저장 -> {paths['transcript_pkl']}")


if __name__ == "__main__":
    main()
