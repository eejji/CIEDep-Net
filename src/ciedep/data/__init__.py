"""데이터셋 구축: 참가자 음성 추출 -> 턴 분리 -> 증강 -> 샘플 인덱스."""

from .augment import augment_minority_classes
from .dataset import (
    CIEDepDataset,
    Sample,
    align_participant_level,
    build_sample_index,
    estimate_pad_length,
    make_loader,
    pad_or_truncate,
    scale_expression_features,
    stack_features,
    stratified_folds,
    train_test_indices,
)
from .labels import build_label_lookup, load_labels, parse_participant_id, phq_score_to_class, scores_to_classes
from .participant_audio import build_participant_dataset, extract_participant_audio
from .turns import extract_turn_pairs, make_dialogue_transcript, save_turn_audio

__all__ = [
    "Sample",
    "CIEDepDataset",
    "build_sample_index",
    "stack_features",
    "align_participant_level",
    "pad_or_truncate",
    "estimate_pad_length",
    "train_test_indices",
    "stratified_folds",
    "scale_expression_features",
    "make_loader",
    "load_labels",
    "build_label_lookup",
    "scores_to_classes",
    "phq_score_to_class",
    "parse_participant_id",
    "extract_participant_audio",
    "build_participant_dataset",
    "extract_turn_pairs",
    "save_turn_audio",
    "make_dialogue_transcript",
    "augment_minority_classes",
]
