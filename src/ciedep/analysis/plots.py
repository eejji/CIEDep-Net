"""결과 시각화: 산점도, 분포, 혼동행렬, 단계별 특징 t-SNE."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..data.labels import phq_score_to_class
from ..metrics import regression_metrics


def _save_or_show(fig, save_path: Optional[str]) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_prediction_scatter(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    save_path: Optional[str] = None,
    xlabel: str = "True PHQ-8 score",
    ylabel: str = "Predicted score",
) -> Dict[str, float]:
    """회귀선 + y=x 참조선 + 지표 텍스트가 있는 산점도."""
    metrics = regression_metrics(y_true, y_pred)

    fig = plt.figure(figsize=(7, 7))
    sns.regplot(
        x=np.asarray(y_true, dtype=float),
        y=np.asarray(y_pred, dtype=float),
        scatter_kws={"alpha": 0.8, "s": 70, "color": "#0052cc", "edgecolor": "black"},
        line_kws={"color": "red", "linestyle": "--", "linewidth": 3},
    )

    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, color="black", linestyle=":", linewidth=2)

    text = f"CCC = {metrics['CCC']:.2f}\nMAE = {metrics['MAE']:.2f}\nr = {metrics['r']:.2f}"
    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=18,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        fontweight="bold",
    )

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linewidth=1.5, alpha=0.6)
    plt.tight_layout()

    _save_or_show(fig, save_path)
    return metrics


def plot_score_distribution(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    save_path: Optional[str] = None,
) -> None:
    """실제/예측 점수 분포 비교 (KDE)."""
    fig = plt.figure(figsize=(8, 6))
    sns.kdeplot(np.asarray(y_true, dtype=float), label="True", fill=True, color="#006400", alpha=0.7, linewidth=2.5)
    sns.kdeplot(np.asarray(y_pred, dtype=float), label="Predicted", fill=True, color="#FF6600", alpha=0.7, linewidth=2.5)
    plt.xlabel("PHQ-8 score", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linewidth=1.5, alpha=0.6)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrix(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    class_bins: Sequence[float] = (9, 14),
    class_names: Sequence[str] = ("Non", "Moderate", "Severe"),
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """회귀 예측을 PHQ-8 3구간으로 변환한 뒤 혼동행렬을 그린다."""
    from sklearn.metrics import confusion_matrix

    true_cls = [phq_score_to_class(v, class_bins) for v in y_true]
    pred_cls = [phq_score_to_class(v, class_bins) for v in y_pred]

    cm = confusion_matrix(true_cls, pred_cls, labels=list(range(len(class_names))))
    display = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm

    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        display,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar=True,
        annot_kws={"size": 16},
    )
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return cm


def plot_stage_tsne(
    features: Dict[str, np.ndarray],
    labels: Sequence[int],
    class_names: Sequence[str] = ("Non", "Moderate", "Severe"),
    perplexity: float = 30.0,
    seed: int = 128,
    save_path: Optional[str] = None,
) -> None:
    """단계별 중간 표현(CIEDepNet.extract_features 결과)을 t-SNE 로 비교한다."""
    from sklearn.manifold import TSNE

    names = list(features.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    labels = np.asarray(labels)

    for ax, name in zip(axes[0], names):
        matrix = np.asarray(features[name], dtype=float)
        embedded = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5.0, (len(matrix) - 1) / 3)),
            random_state=seed,
            init="pca",
        ).fit_transform(matrix)

        for cls, cls_name in enumerate(class_names):
            mask = labels == cls
            if mask.any():
                ax.scatter(embedded[mask, 0], embedded[mask, 1], s=25, alpha=0.75, label=cls_name)

        ax.set_title(name, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0][-1].legend(fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_loss_curve(history: Dict[str, Sequence[float]], fold: int = 1, save_path: Optional[str] = None) -> None:
    """fold 학습 곡선."""
    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title(f"Loss curve - Fold {fold}", fontsize=15)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, save_path)
