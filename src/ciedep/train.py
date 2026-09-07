"""5-fold 교차검증 학습 (논문 IV-D).

Huber loss + AdamW(lr 1e-4, weight decay 5e-5),
Warm-up(10 epoch) 후 CosineAnnealingLR, val loss 기준 early stopping(patience 15),
seed 128 고정. 논문은 단일 H100(80GB)에서 batch size 64 로 학습했다.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import Config
from .data import CIEDepDataset, make_loader, scale_expression_features, stratified_folds, train_test_indices
from .metrics import format_metrics, regression_metrics
from .models import AblationFlags, build_model
from .pipeline import Bundle, load_bundle, shuffle_bundle, subset
from .utils import get_device, set_seed


class EarlyStopping:
    """검증 손실이 patience 동안 개선되지 않으면 학습을 멈춘다."""

    def __init__(self, patience: int = 15, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def build_scheduler(optimizer, warmup_epochs: int, num_epochs: int, eta_min: float):
    """Warm-up(LinearLR) -> CosineAnnealingLR 순차 스케줄러."""
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _run_epoch(model, loader, criterion, device, optimizer=None) -> Dict[str, float]:
    """한 epoch 학습(optimizer 있음) 또는 평가(없음)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, preds, trues = 0.0, [], []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for mel, score, summary, expr, target in loader:
            mel, score = mel.to(device), score.to(device)
            summary, expr, target = summary.to(device), expr.to(device), target.to(device)

            if training:
                optimizer.zero_grad()

            prediction = model(mel, score, summary, expr, return_attn=False)
            loss = criterion(prediction, target)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * mel.size(0)
            preds.extend(prediction.detach().cpu().numpy().ravel())
            trues.extend(target.detach().cpu().numpy().ravel())

    metrics = regression_metrics(trues, preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_fold(
    cfg: Config,
    bundle: Bundle,
    fold: int,
    train_idx: List[int],
    val_idx: List[int],
    ablation: Optional[AblationFlags] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """한 fold 를 학습하고 최고 성능 체크포인트를 저장한다."""
    train_cfg = cfg["train"]
    device = device or get_device()

    mel_tr, score_tr, summary_tr, expr_tr, y_tr = subset(bundle, train_idx)
    mel_va, score_va, summary_va, expr_va, y_va = subset(bundle, val_idx)

    # 표현 특징은 fold 학습 데이터에만 맞춘 StandardScaler 로 정규화한다.
    if train_cfg.get("scale_expression_feature", True):
        expr_tr, (expr_va,), _ = scale_expression_features(expr_tr, [expr_va])

    train_loader = make_loader(
        CIEDepDataset(mel_tr, score_tr, summary_tr, expr_tr, y_tr), train_cfg["batch_size"], shuffle=True
    )
    val_loader = make_loader(
        CIEDepDataset(mel_va, score_va, summary_va, expr_va, y_va), train_cfg["batch_size"], shuffle=False
    )

    model = build_model(dict(cfg["model"]), ablation).to(device)
    criterion = nn.HuberLoss() if train_cfg.get("loss", "huber") == "huber" else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(
        optimizer, train_cfg["warmup_epochs"], train_cfg["num_epochs"], train_cfg["eta_min"]
    )
    stopper = EarlyStopping(patience=train_cfg["early_stopping_patience"])

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ciedep_fold{fold + 1}.pth"

    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_mae"].append(train_metrics["MAE"])
        history["val_mae"].append(val_metrics["MAE"])

        if verbose:
            print(
                f"Fold {fold + 1} | Epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} MAE {train_metrics['MAE']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} MAE {val_metrics['MAE']:.4f}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), ckpt_path)
            if verbose:
                print(f"  -> best model saved (val loss {best_val_loss:.4f})")

        stopper(val_metrics["loss"])
        if stopper.early_stop:
            if verbose:
                print("  -> early stopping")
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold + 1,
        "best_val_loss": best_val_loss,
        "checkpoint": str(ckpt_path),
        "history": history,
    }


def evaluate_checkpoint(
    cfg: Config,
    bundle: Bundle,
    checkpoint: str,
    indices: List[int],
    ablation: Optional[AblationFlags] = None,
    device: Optional[torch.device] = None,
    scaler_reference: Optional[List[int]] = None,
) -> Dict[str, float]:
    """저장된 체크포인트로 지정한 인덱스 집합을 평가한다."""
    device = device or get_device()

    mel, score, summary, expr, target = subset(bundle, indices)
    if cfg["train"].get("scale_expression_feature", True) and scaler_reference:
        ref_expr = bundle.expr[list(scaler_reference)]
        _, (expr,), _ = scale_expression_features(ref_expr, [expr])

    loader = make_loader(CIEDepDataset(mel, score, summary, expr, target), cfg["train"]["batch_size"])

    model = build_model(dict(cfg["model"]), ablation).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for mel_b, score_b, summary_b, expr_b, target_b in loader:
            prediction = model(
                mel_b.to(device), score_b.to(device), summary_b.to(device), expr_b.to(device), return_attn=False
            )
            preds.extend(prediction.cpu().numpy().ravel())
            trues.extend(target_b.numpy().ravel())

    return regression_metrics(trues, preds)


def run_cross_validation(
    cfg: Config,
    ablation: Optional[AblationFlags] = None,
    tag: str = "proposed",
    verbose: bool = True,
) -> Dict[str, object]:
    """논문 IV-D 의 평가 절차: 20% 테스트 고정 후 나머지로 5-fold 교차검증."""
    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"device: {device}")

    bundle = shuffle_bundle(load_bundle(cfg, verbose=verbose), seed=cfg["train"]["seed"])

    train_val_idx, test_idx = train_test_indices(
        bundle.labels, test_size=cfg["train"]["test_size"], seed=cfg["train"]["seed"]
    )
    print(f"전체 {len(bundle)} | train+val {len(train_val_idx)} | test {len(test_idx)}")

    fold_records, test_metrics = [], []
    for fold, tr_idx, va_idx in stratified_folds(
        train_val_idx, bundle.labels, n_splits=cfg["train"]["n_splits"], seed=cfg["train"]["seed"]
    ):
        record = train_fold(cfg, bundle, fold, tr_idx, va_idx, ablation, device, verbose)
        metrics = evaluate_checkpoint(
            cfg, bundle, record["checkpoint"], test_idx, ablation, device, scaler_reference=tr_idx
        )
        record["test_metrics"] = metrics
        fold_records.append(record)
        test_metrics.append(metrics)
        print(f"[fold {fold + 1}] test  {format_metrics(metrics)}")

    summary = {
        key: {
            "mean": float(np.mean([m[key] for m in test_metrics])),
            "std": float(np.std([m[key] for m in test_metrics])),
        }
        for key in ("MAE", "RMSE", "CCC", "r", "R2")
    }

    print("\n=== 5-fold 테스트 성능 ===")
    for key, value in summary.items():
        print(f"{key:5s}: {value['mean']:.4f} +- {value['std']:.4f}")

    result = {
        "tag": tag,
        "dataset": cfg["dataset"]["name"],
        "ablation": asdict(ablation) if ablation else asdict(AblationFlags()),
        "folds": [{k: v for k, v in r.items() if k != "history"} for r in fold_records],
        "summary": summary,
    }

    result_dir = Path(cfg["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    return result
