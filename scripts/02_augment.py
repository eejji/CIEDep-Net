"""2단계 - 소수 클래스 데이터 증강.

PHQ-8 3구간 중 moderate(10-14) / severe(15-24) 에만 pitch shift, time stretch 를 적용한다
(논문 IV-B). time shift 는 시계열 순서를 왜곡하므로 사용하지 않는다.

    python scripts/02_augment.py --config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ciedep.config import load_config
from ciedep.data import augment_minority_classes, build_label_lookup, load_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="소수 클래스 오디오 증강")
    parser.add_argument("--config", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg, paths, aug_cfg = cfg["dataset"], cfg["paths"], cfg["augment"]

    label_df = load_labels(
        dataset_cfg["label_csv"], dataset_cfg["label_columns"], dataset_cfg.get("excluded_indices", ())
    )
    label_lookup = build_label_lookup(label_df)

    result = augment_minority_classes(
        label_lookup=label_lookup,
        source_dir=paths["participant_dir"],
        moderate_dir=paths["augment_moderate_dir"],
        severe_dir=paths["augment_severe_dir"],
        moderate_extra=aug_cfg["moderate_extra"],
        severe_extra=aug_cfg["severe_extra"],
        sample_rate=cfg["audio"]["sample_rate"],
        seed=cfg["train"]["seed"],
        pitch_range=aug_cfg["pitch_shift_range"],
        pitch_min_gap=aug_cfg["pitch_min_gap"],
        stretch_range=aug_cfg["time_stretch_range"],
        stretch_min_gap=aug_cfg["time_min_gap"],
        verbose=not args.quiet,
    )

    print(f"\nmoderate {len(result['moderate'])}개, severe {len(result['severe'])}개 생성")


if __name__ == "__main__":
    main()
