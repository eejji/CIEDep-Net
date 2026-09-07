"""설정 -> 학습 가능한 배열 묶음.

특징 파일과 LLM 산출물을 읽어 샘플 순서에 맞춰 정렬한다.

기대하는 파일 (모두 pickle):
  <feature_dir>/mel_<window>s.pkl              {sample_id: (T, 80)}
  <feature_dir>/<expression_model>_<window>s.pkl  {sample_id: (T, D)}
  <llm_dir>/<model>_score.pkl                  {participant_id: float}  (0~1)
  <llm_dir>/<model>_summary_embedding.pkl      {participant_id: (768,)}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .data import (
    align_participant_level,
    build_label_lookup,
    build_sample_index,
    estimate_pad_length,
    load_labels,
    scores_to_classes,
    stack_features,
)
from .data.dataset import Sample
from .utils import load_pickle


def feature_filename(kind: str, window_sec: float) -> str:
    """특징 파일명 규칙: '<kind>_<window>s.pkl' (예: mel_4s.pkl, wav2vec2_4s.pkl)."""
    window = int(window_sec) if float(window_sec).is_integer() else window_sec
    return f"{kind}_{window}s.pkl"


def llm_filename(model_id: str, kind: str) -> str:
    """LLM 산출물 파일명 규칙: '<모델명>_<kind>.pkl'."""
    return f"{model_id.split('/')[-1]}_{kind}.pkl"


@dataclass
class Bundle:
    """모델 입력으로 정렬이 끝난 배열 묶음."""

    samples: List[Sample]
    mel: np.ndarray          # (N, T, 80)
    expr: np.ndarray         # (N, T, D)
    score: np.ndarray        # (N, 1)   LLM 우울증 점수 (0~1)
    summary: np.ndarray      # (N, 768) LLM 내면 요약 임베딩
    target: np.ndarray       # (N, 1)   실제 PHQ-8
    labels: np.ndarray       # (N,)     층화용 3구간 클래스
    pad_length: int

    def __len__(self) -> int:
        return len(self.samples)


def load_bundle(cfg: Config, verbose: bool = True) -> Bundle:
    """설정을 읽어 학습에 바로 쓸 수 있는 Bundle 을 만든다."""
    dataset_cfg, paths, audio_cfg = cfg["dataset"], cfg["paths"], cfg["audio"]

    # 1) 라벨
    label_df = load_labels(
        dataset_cfg["label_csv"],
        dataset_cfg["label_columns"],
        dataset_cfg.get("excluded_indices", ()),
    )
    label_lookup = build_label_lookup(label_df)

    # 2) 샘플 인덱스 (원본 + 증강)
    samples = build_sample_index(
        paths["participant_dir"],
        [paths["augment_moderate_dir"], paths["augment_severe_dir"]],
        label_lookup,
        class_bins=cfg["augment"]["class_bins"],
        verbose=verbose,
    )
    if not samples:
        raise RuntimeError("샘플이 하나도 없습니다. configs 의 경로를 확인하세요.")

    # 3) 특징
    feature_dir = Path(paths["feature_dir"])
    window_sec = audio_cfg["window_sec"]

    mel_dict: Dict[str, np.ndarray] = load_pickle(feature_dir / feature_filename("mel", window_sec))
    expr_name = cfg["features"]["expression_model"]
    expr_dict: Dict[str, np.ndarray] = load_pickle(feature_dir / feature_filename(expr_name, window_sec))

    pad_length = audio_cfg.get("pad_length") or estimate_pad_length(mel_dict)

    mel = stack_features(mel_dict, samples, pad_length)
    expr = stack_features(expr_dict, samples, pad_length)

    # 4) LLM 산출물 (참가자 단위 -> 샘플 단위로 확장)
    llm_dir = Path(paths["llm_dir"])
    model_id = cfg["llm"]["model_id"]

    score_dict = load_pickle(llm_dir / llm_filename(model_id, "score"))
    summary_dict = load_pickle(llm_dir / llm_filename(model_id, "summary_embedding"))

    score = align_participant_level(score_dict, samples).reshape(-1, 1)
    summary = align_participant_level(summary_dict, samples)

    # 5) 정답 라벨
    target = np.array([s.phq_score for s in samples], dtype=np.float32).reshape(-1, 1)
    labels = scores_to_classes([s.phq_score for s in samples], cfg["augment"]["class_bins"])

    if verbose:
        print(
            f"mel {mel.shape} | expr {expr.shape} | score {score.shape} | "
            f"summary {summary.shape} | target {target.shape} | pad_length={pad_length}"
        )

    return Bundle(
        samples=samples,
        mel=mel,
        expr=expr,
        score=score,
        summary=summary,
        target=target,
        labels=labels,
        pad_length=pad_length,
    )


def shuffle_bundle(bundle: Bundle, seed: int = 128) -> Bundle:
    """원본/증강이 뭉쳐 있지 않도록 전체 샘플 순서를 섞는다."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bundle))
    return Bundle(
        samples=[bundle.samples[i] for i in order],
        mel=bundle.mel[order],
        expr=bundle.expr[order],
        score=bundle.score[order],
        summary=bundle.summary[order],
        target=bundle.target[order],
        labels=bundle.labels[order],
        pad_length=bundle.pad_length,
    )


def subset(bundle: Bundle, indices) -> Tuple[np.ndarray, ...]:
    """인덱스로 (mel, score, summary, expr, target) 을 잘라 낸다."""
    idx = list(indices)
    return (
        bundle.mel[idx],
        bundle.score[idx],
        bundle.summary[idx],
        bundle.expr[idx],
        bundle.target[idx],
    )
