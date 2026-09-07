"""0단계(선택) - 기존 노트북 산출물을 이 저장소 형식으로 변환.

이미 GPU로 뽑아 둔 특징과 LLM 결과가 있다면 재추출 없이 그대로 쓸 수 있다.

변환 규칙
  특징 : {'Non_depression': [...], 'Moderate': [...], 'Severe': [...]}  (디렉터리 나열 순서)
         -> {sample_id: (T, D)}
  LLM  : [score, ...] / [summary, ...] / [embedding, ...]  (라벨 CSV 행 순서)
         -> {participant_id: value}

    python scripts/00_migrate_legacy.py --config configs/daic_woz.yaml \
        --legacy-feature "D:/__Journal_Audio/Expression_feature/wav2vec2_4_base_960h.pkl" --kind wav2vec2
    python scripts/00_migrate_legacy.py --config configs/daic_woz.yaml \
        --legacy-llm-score "D:/__Journal_Audio/_Review/LLM_data/Qwen2.5-7B-Instruct_score.pkl" \
        --legacy-llm-summary "D:/__Journal_Audio/_Review/LLM_data/Qwen2.5-7B-Instruct_summary.pkl" \
        --legacy-llm-embedding "D:/.../summary_embedding/Qwen2.5-7B-Instruct_summary_embedding.pkl"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

import _bootstrap  # noqa: F401

from ciedep.config import load_config
from ciedep.data import load_labels
from ciedep.pipeline import feature_filename, llm_filename
from ciedep.utils import load_pickle, save_pickle

LEGACY_KEYS = ["Non_depression", "Moderate", "Severe"]


def migrate_features(cfg, legacy_path: str, kind: str) -> Path:
    """3클래스 리스트 형식 특징을 {sample_id: array} 로 바꾼다."""
    paths, audio_cfg = cfg["paths"], cfg["audio"]
    legacy: Dict[str, list] = load_pickle(legacy_path)

    missing = [k for k in LEGACY_KEYS if k not in legacy]
    if missing:
        raise SystemExit(f"기대한 키가 없습니다: {missing} (있는 키: {list(legacy)})")

    directories = [paths["participant_dir"], paths["augment_moderate_dir"], paths["augment_severe_dir"]]

    converted: Dict[str, np.ndarray] = {}
    for key, directory in zip(LEGACY_KEYS, directories):
        wav_files = sorted(Path(directory).glob("*.wav"))
        values = legacy[key]
        if len(wav_files) != len(values):
            raise SystemExit(
                f"[{key}] 파일 수와 특징 수가 다릅니다: {directory} 에 {len(wav_files)}개, "
                f"특징 {len(values)}개.\n"
                "기존 노트북이 사용한 디렉터리를 configs 에 맞춰 주세요."
            )
        for wav, value in zip(wav_files, values):
            converted[wav.stem] = np.asarray(value, dtype=np.float32)

    out_path = Path(paths["feature_dir"]) / feature_filename(kind, audio_cfg["window_sec"])
    save_pickle(converted, out_path)
    print(f"특징 {len(converted)}개 변환 -> {out_path}")
    return out_path


def migrate_llm(cfg, score_path: str | None, summary_path: str | None, embedding_path: str | None) -> None:
    """라벨 CSV 행 순서를 가정한 리스트를 {participant_id: value} 로 바꾼다."""
    dataset_cfg, paths, llm_cfg = cfg["dataset"], cfg["paths"], cfg["llm"]

    label_df = load_labels(
        dataset_cfg["label_csv"], dataset_cfg["label_columns"], dataset_cfg.get("excluded_indices", ())
    )
    participant_ids: List[str] = label_df["participant_id"].tolist()
    llm_dir = Path(paths["llm_dir"])
    model_id = llm_cfg["model_id"]

    for legacy_path, kind in [
        (score_path, "score"),
        (summary_path, "summary"),
        (embedding_path, "summary_embedding"),
    ]:
        if not legacy_path:
            continue

        values = load_pickle(legacy_path)
        values = list(values)
        if len(values) != len(participant_ids):
            raise SystemExit(
                f"[{kind}] 길이가 맞지 않습니다: 참가자 {len(participant_ids)}명, 값 {len(values)}개.\n"
                "dataset.excluded_indices 설정을 확인하세요."
            )

        converted = {pid: value for pid, value in zip(participant_ids, values)}
        out_path = llm_dir / llm_filename(model_id, kind)
        save_pickle(converted, out_path)
        print(f"{kind} {len(converted)}건 변환 -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="기존 노트북 산출물 변환")
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-feature", default=None, help="3클래스 리스트 형식 특징 pkl")
    parser.add_argument("--kind", default="wav2vec2", help="특징 종류 (mel, wav2vec2, hubert, ...)")
    parser.add_argument("--legacy-llm-score", default=None)
    parser.add_argument("--legacy-llm-summary", default=None)
    parser.add_argument("--legacy-llm-embedding", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.legacy_feature:
        migrate_features(cfg, args.legacy_feature, args.kind)

    if args.legacy_llm_score or args.legacy_llm_summary or args.legacy_llm_embedding:
        migrate_llm(cfg, args.legacy_llm_score, args.legacy_llm_summary, args.legacy_llm_embedding)

    if not any([args.legacy_feature, args.legacy_llm_score, args.legacy_llm_summary, args.legacy_llm_embedding]):
        parser.error("변환할 대상을 하나 이상 지정하세요.")


if __name__ == "__main__":
    main()
