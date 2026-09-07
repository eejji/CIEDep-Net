"""7단계 - 저장된 체크포인트로 테스트셋 평가 및 시각화.

    python scripts/07_evaluate.py --config configs/daic_woz.yaml
    python scripts/07_evaluate.py --config configs/daic_woz.yaml --plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from ciedep.analysis import plot_confusion_matrix, plot_prediction_scatter, plot_score_distribution
from ciedep.config import load_config
from ciedep.data import CIEDepDataset, make_loader, scale_expression_features, stratified_folds, train_test_indices
from ciedep.metrics import format_metrics, regression_metrics
from ciedep.models import AblationFlags, build_model
from ciedep.pipeline import load_bundle, shuffle_bundle, subset
from ciedep.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="테스트셋 평가")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", default="proposed")
    parser.add_argument("--plots", action="store_true", help="산점도/분포/혼동행렬 저장")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device()

    bundle = shuffle_bundle(load_bundle(cfg), seed=cfg["train"]["seed"])
    train_val_idx, test_idx = train_test_indices(
        bundle.labels, test_size=cfg["train"]["test_size"], seed=cfg["train"]["seed"]
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    result_dir = Path(cfg["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    fold_train_indices = {
        fold: tr_idx
        for fold, tr_idx, _ in stratified_folds(
            train_val_idx, bundle.labels, n_splits=cfg["train"]["n_splits"], seed=cfg["train"]["seed"]
        )
    }

    all_metrics, fold_predictions = [], []
    for fold in range(cfg["train"]["n_splits"]):
        ckpt = ckpt_dir / f"ciedep_fold{fold + 1}.pth"
        if not ckpt.exists():
            print(f"[skip] 체크포인트 없음: {ckpt}")
            continue

        mel, score, summary, expr, target = subset(bundle, test_idx)
        if cfg["train"].get("scale_expression_feature", True):
            ref = bundle.expr[list(fold_train_indices[fold])]
            _, (expr,), _ = scale_expression_features(ref, [expr])

        loader = make_loader(CIEDepDataset(mel, score, summary, expr, target), cfg["train"]["batch_size"])

        model = build_model(
            dict(cfg["model"]),
            AblationFlags(expression_backbone=cfg["model"]["expression_backbone"]),
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for mel_b, score_b, summary_b, expr_b, target_b in loader:
                out = model(
                    mel_b.to(device), score_b.to(device), summary_b.to(device), expr_b.to(device), return_attn=False
                )
                preds.extend(out.cpu().numpy().ravel())
                trues.extend(target_b.numpy().ravel())

        metrics = regression_metrics(trues, preds)
        all_metrics.append(metrics)
        fold_predictions.append(np.array(preds))
        print(f"[fold {fold + 1}] {format_metrics(metrics)}")

    if not all_metrics:
        raise SystemExit("평가할 체크포인트가 없습니다. 먼저 scripts/06_train.py 를 실행하세요.")

    summary_stats = {
        key: {
            "mean": float(np.mean([m[key] for m in all_metrics])),
            "std": float(np.std([m[key] for m in all_metrics])),
        }
        for key in ("MAE", "RMSE", "CCC", "r", "R2")
    }

    print("\n=== 테스트 성능 (5-fold 평균) ===")
    for key, value in summary_stats.items():
        print(f"{key:5s}: {value['mean']:.4f} +- {value['std']:.4f}")

    out_path = result_dir / f"{args.tag}_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"folds": all_metrics, "summary": summary_stats}, f, indent=2)
    print(f"\n결과 저장: {out_path}")

    if args.plots:
        y_true = bundle.target[list(test_idx)].ravel()
        y_pred = np.mean(fold_predictions, axis=0)     # fold 앙상블 평균

        plot_prediction_scatter(y_true, y_pred, save_path=str(result_dir / f"{args.tag}_scatter.png"))
        plot_score_distribution(y_true, y_pred, save_path=str(result_dir / f"{args.tag}_distribution.png"))
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_bins=cfg["augment"]["class_bins"],
            save_path=str(result_dir / f"{args.tag}_confusion.png"),
        )
        print(f"그림 저장 -> {result_dir}")


if __name__ == "__main__":
    main()
