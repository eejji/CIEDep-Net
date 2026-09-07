"""5단계 - 특징 추출.

  mel        : Cognition 입력 (80-bin, 세그먼트별 시간축 평균, 참가자 단위 min-max 정규화)
  wav2vec2 외 : Expression 입력 (SSL 인코더의 프레임 평균)

두 특징 모두 4초 윈도우 / 1초 중첩으로 세그먼트를 만든 뒤 계산한다 (논문 IV-B, Table I).
결과는 <feature_dir>/<kind>_<window>s.pkl 에 {sample_id: (T, D)} 형태로 저장된다.

    python scripts/05_extract_features.py --config configs/daic_woz.yaml --kind mel
    python scripts/05_extract_features.py --config configs/daic_woz.yaml --kind wav2vec2
    python scripts/05_extract_features.py --config configs/daic_woz.yaml --kind all --window 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from ciedep.config import load_config
from ciedep.features import DEFAULT_MODEL_IDS, SSLFeatureExtractor, build_mel_transform, extract_mel_directory
from ciedep.pipeline import feature_filename
from ciedep.utils import save_pickle

SSL_KINDS = list(DEFAULT_MODEL_IDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="인지/표현 특징 추출")
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", default="all", choices=["all", "mel", *SSL_KINDS])
    parser.add_argument("--window", type=float, default=None, help="윈도우 길이(초). 기본은 설정값")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, audio_cfg = cfg["paths"], cfg["audio"]

    window_sec = args.window or audio_cfg["window_sec"]
    hop_sec = window_sec * (audio_cfg["hop_sec"] / audio_cfg["window_sec"])  # 중첩 비율 유지

    audio_dirs = [paths["participant_dir"], paths["augment_moderate_dir"], paths["augment_severe_dir"]]
    feature_dir = Path(paths["feature_dir"])
    feature_dir.mkdir(parents=True, exist_ok=True)

    kinds = ["mel", cfg["features"]["expression_model"]] if args.kind == "all" else [args.kind]

    for kind in kinds:
        print(f"\n=== {kind} (window {window_sec}s, hop {hop_sec}s) ===")

        if kind == "mel":
            transform = build_mel_transform(
                sample_rate=audio_cfg["sample_rate"],
                n_mels=audio_cfg["n_mels"],
                frame_length_sec=audio_cfg["frame_length_sec"],
                frame_stride_sec=audio_cfg["frame_stride_sec"],
            )
            features = extract_mel_directory(
                audio_dirs,
                transform,
                sample_rate=audio_cfg["sample_rate"],
                window_sec=window_sec,
                hop_sec=hop_sec,
                top_db=audio_cfg["top_db"],
                verbose=not args.quiet,
            )
        else:
            extractor = SSLFeatureExtractor(
                model_name=kind, model_id=cfg["features"]["ssl_model_ids"].get(kind)
            )
            features = extractor.extract_directories(
                audio_dirs,
                sample_rate=audio_cfg["sample_rate"],
                window_sec=window_sec,
                hop_sec=hop_sec,
                verbose=not args.quiet,
            )

        out_path = feature_dir / feature_filename(kind, window_sec)
        save_pickle(features, out_path)
        print(f"{len(features)}개 샘플 저장 -> {out_path}")


if __name__ == "__main__":
    main()
