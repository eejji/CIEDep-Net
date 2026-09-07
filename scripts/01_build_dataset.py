"""1단계 - 데이터셋 구축.

원본 배포본의 전사 시간 정보로 참가자 발화만 추출해 하나의 음성 파일로 잇는다.
면담자 음성은 우울증 분석 편향을 막기 위해 모두 제거한다 (논문 IV-B).

    python scripts/01_build_dataset.py --config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ciedep.config import load_config
from ciedep.data import build_participant_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="참가자 전용 음성 생성")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg, paths = cfg["dataset"], cfg["paths"]

    saved = build_participant_dataset(
        raw_dir=dataset_cfg["raw_dir"],
        output_dir=paths["participant_dir"],
        columns=dataset_cfg["transcript_columns"],
        participant_label=dataset_cfg["participant_label"],
        sample_rate=cfg["audio"]["sample_rate"],
        audio_suffix=dataset_cfg["audio_suffix"],
        transcript_suffix=dataset_cfg["transcript_suffix"],
        verbose=not args.quiet,
    )

    print(f"\n참가자 음성 {len(saved)}개 저장 -> {paths['participant_dir']}")


if __name__ == "__main__":
    main()
