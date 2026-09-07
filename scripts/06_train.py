"""6단계 - CIEDep-Net 학습 및 5-fold 교차검증.

    python scripts/06_train.py --config configs/daic_woz.yaml
    python scripts/06_train.py --config configs/e_daic.yaml --tag proposed

Ablation (논문 Table III / IV):
    python scripts/06_train.py --config configs/daic_woz.yaml --tag wo_score      --no-score
    python scripts/06_train.py --config configs/daic_woz.yaml --tag wo_summary    --no-summary
    python scripts/06_train.py --config configs/daic_woz.yaml --tag wo_cognition  --no-cognition
    python scripts/06_train.py --config configs/daic_woz.yaml --tag baseline3     --fusion concat
    python scripts/06_train.py --config configs/daic_woz.yaml --tag baseline4     --expression-backbone transformer
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ciedep.config import load_config
from ciedep.models import AblationFlags
from ciedep.train import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="CIEDep-Net 5-fold 학습")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", default="proposed", help="결과 파일 이름")

    parser.add_argument("--no-cognition", action="store_true", help="Cognition stage 제거")
    parser.add_argument("--no-interpretation", action="store_true", help="Interpretation stage 제거")
    parser.add_argument("--no-expression", action="store_true", help="Expression 특징 융합 제거")
    parser.add_argument("--no-score", action="store_true", help="LLM 우울증 점수 제거")
    parser.add_argument("--no-summary", action="store_true", help="LLM 내면 요약 제거")
    parser.add_argument("--fusion", default=None, choices=["sc_ca", "concat"])
    parser.add_argument("--expression-backbone", default=None, choices=["conformer", "transformer"])

    parser.add_argument("--epochs", type=int, default=None, help="설정의 num_epochs 를 덮어쓴다")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["num_epochs"] = args.epochs

    ablation = AblationFlags(
        use_cognition=not args.no_cognition,
        use_interpretation=not args.no_interpretation,
        use_expression=not args.no_expression,
        use_score=not args.no_score,
        use_summary=not args.no_summary,
        fusion=args.fusion or cfg["model"].get("fusion", "sc_ca"),
        expression_backbone=args.expression_backbone or cfg["model"]["expression_backbone"],
    )

    run_cross_validation(cfg, ablation=ablation, tag=args.tag, verbose=not args.quiet)


if __name__ == "__main__":
    main()
