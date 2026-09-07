"""공통 유틸: 시드 고정, pickle I/O, 디바이스 선택."""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 128) -> None:
    """논문 재현을 위한 전역 시드 고정 (seed=128)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_pickle(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | os.PathLike) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def drop_indices(items: list, indices) -> list:
    """논문에서 제외한 참가자 인덱스를 리스트에서 제거한다."""
    excluded = set(indices or [])
    return [v for i, v in enumerate(items) if i not in excluded]
