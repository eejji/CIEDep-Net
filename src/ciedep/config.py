"""YAML 설정 로더.

모든 경로/하이퍼파라미터는 configs/*.yaml 에 있다. 코드에는 경로를 두지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """점 표기법으로 접근 가능한 dict (cfg.model.d_input)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | os.PathLike) -> Config:
    """YAML 설정 파일을 읽어 Config 로 반환한다."""
    with open(path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return Config(raw)


def resolve(path: str | os.PathLike, create: bool = False) -> Path:
    """경로 문자열을 Path 로 정규화한다. create=True 면 디렉터리를 만든다."""
    p = Path(path).expanduser()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p
