"""4단계 - Interpretation stage: LLM 우울증 점수 + 내면 요약 생성.

CoT 프롬프트와 self-consistency(5회 생성)를 적용한다 (논문 III-B-2).
  - 점수 : 5개 값의 평균
  - 요약 : 5개 임베딩의 평균에 가장 가까운 출력

    python scripts/04_llm_interpretation.py --config configs/daic_woz.yaml
    python scripts/04_llm_interpretation.py --config configs/daic_woz.yaml --strategy cot
    python scripts/04_llm_interpretation.py --config configs/daic_woz.yaml --model Qwen/Qwen2.5-7B-Instruct

인증이 필요한 모델은 환경변수 HF_TOKEN 을 설정해 두세요 (코드에 토큰을 넣지 않습니다).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from ciedep.config import load_config
from ciedep.features.text_embed import embed_summaries
from ciedep.llm import (
    LLMRunner,
    build_prompt,
    parse_depression_score,
    parse_inner_summary,
    run_self_consistency_score,
    run_self_consistency_summary,
    self_validation_prompt,
)
from ciedep.pipeline import llm_filename
from ciedep.utils import load_pickle, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 기반 해석 특징 생성")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="설정의 llm.model_id 를 덮어쓴다")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["standard", "cot", "cot_self_consistency", "cot_self_validation"],
        help="설정의 llm.prompt_strategy 를 덮어쓴다",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm_cfg, paths = cfg["llm"], cfg["paths"]

    model_id = args.model or llm_cfg["model_id"]
    strategy = args.strategy or llm_cfg["prompt_strategy"]
    print(f"모델: {model_id} | 전략: {strategy}")

    dialogues = load_pickle(paths["transcript_pkl"])
    if isinstance(dialogues, (list, tuple)):
        raise SystemExit(
            "전사 파일이 리스트입니다. {participant_id: dialogue} 형태의 dict 로 저장해 주세요 "
            "(scripts/03_transcribe.py 참고)."
        )

    participant_ids = list(dialogues.keys())
    if args.limit:
        participant_ids = participant_ids[: args.limit]

    runner = LLMRunner(model_id=model_id)
    embed_fn = lambda texts: embed_summaries(texts, model_name=llm_cfg["sentence_embedder"], clean=False)

    scores, summaries = {}, {}

    for i, pid in enumerate(participant_ids, start=1):
        transcript = dialogues[pid]

        score_messages = build_prompt(transcript, strategy, "score")
        summary_messages = build_prompt(transcript, strategy, "summary")

        if strategy == "cot_self_consistency":
            score, _ = run_self_consistency_score(
                runner,
                score_messages,
                n_samples=llm_cfg["n_samples"],
                temperature=llm_cfg["temperature"],
                top_p=llm_cfg["top_p"],
                max_new_tokens=llm_cfg["max_new_tokens_score"],
            )
            summary, _ = run_self_consistency_summary(
                runner,
                summary_messages,
                embed_fn,
                n_samples=llm_cfg["n_samples"],
                temperature=llm_cfg["temperature"],
                top_p=llm_cfg["top_p"],
                max_new_tokens=llm_cfg["max_new_tokens"],
            )
        else:
            score = parse_depression_score(
                runner.generate_one(score_messages, max_new_tokens=llm_cfg["max_new_tokens_score"])
            )
            summary = parse_inner_summary(
                runner.generate_one(summary_messages, max_new_tokens=llm_cfg["max_new_tokens"])
            )

            if strategy == "cot_self_validation":
                score = parse_depression_score(
                    runner.generate_one(
                        self_validation_prompt(transcript, f">>> Depression score: {score}", "score"),
                        max_new_tokens=llm_cfg["max_new_tokens_score"],
                    )
                ) or score
                summary = (
                    parse_inner_summary(
                        runner.generate_one(
                            self_validation_prompt(transcript, summary, "summary"),
                            max_new_tokens=llm_cfg["max_new_tokens"],
                        )
                    )
                    or summary
                )

        scores[pid] = score
        summaries[pid] = summary
        print(f"[{i}/{len(participant_ids)}] {pid}: score={score}")

    failed = [pid for pid, s in scores.items() if s is None]
    if failed:
        print(f"\n[warn] 점수 파싱 실패 {len(failed)}명: {failed[:10]}")

    llm_dir = Path(paths["llm_dir"])
    save_pickle(scores, llm_dir / llm_filename(model_id, "score"))
    save_pickle(summaries, llm_dir / llm_filename(model_id, "summary"))

    # 요약을 문장 임베딩으로 변환해 함께 저장한다 (Interpretation stage 입력).
    valid_ids = [pid for pid in participant_ids if summaries[pid]]
    matrix = embed_summaries(
        [summaries[pid] for pid in valid_ids], model_name=llm_cfg["sentence_embedder"]
    )
    embedding_dict = {pid: matrix[i] for i, pid in enumerate(valid_ids)}
    save_pickle(embedding_dict, llm_dir / llm_filename(model_id, "summary_embedding"))

    print(f"\n저장 완료 -> {llm_dir}")


if __name__ == "__main__":
    main()
