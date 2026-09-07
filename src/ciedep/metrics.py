"""평가 지표: MAE, RMSE, CCC, Pearson r, R^2 (논문 Eq. 17-20)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def concordance_correlation_coefficient(y_true, y_pred) -> float:
    """CCC = 2 * cov / (var_true + var_pred + (mean_true - mean_pred)^2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true = y_true.var(ddof=1)
    var_pred = y_pred.var(ddof=1)
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]

    return float(2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2))


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """논문 Table 들에서 보고하는 전체 지표를 한 번에 계산한다."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    r, p = pearsonr(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "CCC": concordance_correlation_coefficient(y_true, y_pred),
        "r": float(r),
        "p": float(p),
        "R2": float(r2_score(y_true, y_pred)),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    order = ["MAE", "RMSE", "CCC", "r", "R2"]
    return " | ".join(f"{k}={metrics[k]:.4f}" for k in order if k in metrics)
