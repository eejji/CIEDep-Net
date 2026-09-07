"""부가 분석 - LLM 산출물의 예측 타당성 검증 (논문 Abstract, Table II).

생성 우울증 점수와 실제 PHQ-8 의 상관, 내면 요약의 BERTScore 를 계산한다.
논문 보고값: Pearson r = 0.69 (p < 0.01), BERTScore = 0.8

    python scripts/08_llm_validity.py --config configs/daic_woz.yaml
    python scripts/08_llm_validity.py --config configs/daic_woz.yaml --bertscore
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from ciedep.analysis import bertscore_summaries, evaluate_generated_scores
from ciedep.analysis.plots import plot_prediction_scatter
from ciedep.config import load_config
from ciedep.data import build_label_lookup, load_labels
from ciedep.llm import extract_participant_utterances
from ciedep.pipeline import llm_filename
from ciedep.utils import load_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 점수/요약 타당성 검증")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--bertscore", action="store_true", help="내면 요약 BERTScore 계산 (느림)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg, paths, llm_cfg = cfg["dataset"], cfg["paths"], cfg["llm"]
    model_id = args.model or llm_cfg["model_id"]

    label_df = load_labels(
        dataset_cfg["label_csv"], dataset_cfg["label_columns"], dataset_cfg.get("excluded_indices", ())
    )
    label_lookup = build_label_lookup(label_df)

    llm_dir = Path(paths["llm_dir"])
    scores = load_pickle(llm_dir / llm_filename(model_id, "score"))

    pids = [pid for pid, s in scores.items() if s is not None and pid in label_lookup]
    generated = [scores[pid] for pid in pids]
    actual = [label_lookup[pid]["phq_score"] for pid in pids]

    metrics = evaluate_generated_scores(generated, actual, scale=llm_cfg["score_scale"])
    print(f"\n=== 생성 점수 vs 실제 PHQ-8 (n={len(pids)}) ===")
    for key in ("r", "p", "CCC", "MAE", "RMSE", "R2"):
        print(f"{key:5s}: {metrics[key]:.4f}")

    result = {"model": model_id, "n": len(pids), "score_metrics": metrics}

    if args.bertscore:
        summaries = load_pickle(llm_dir / llm_filename(model_id, "summary"))
        dialogues = load_pickle(paths["transcript_pkl"])

        valid = [pid for pid in pids if summaries.get(pid) and pid in dialogues]
        references = [
            " ".join(extract_participant_utterances(dialogues[pid], dataset_cfg["participant_label"]))
            for pid in valid
        ]
        bert = bertscore_summaries([summaries[pid] for pid in valid], references)
        print(f"\nBERTScore F1: {bert['f1_mean']:.4f} +- {bert['f1_std']:.4f} (n={len(valid)})")
        result["bertscore"] = {"f1_mean": bert["f1_mean"], "f1_std": bert["f1_std"], "n": len(valid)}

    result_dir = Path(paths["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"llm_validity_{model_id.split('/')[-1]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    if args.plot:
        predicted = np.array(generated) * llm_cfg["score_scale"]
        plot_prediction_scatter(
            actual,
            predicted,
            save_path=str(result_dir / f"llm_score_{model_id.split('/')[-1]}.png"),
            ylabel="Generated score",
        )


if __name__ == "__main__":
    main()
