"""Prepare instructor annotations, prepare response reviews, or score a completed review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import EvaluationError, load_cases  # noqa: E402

from hrlearnium.eval_tools import annotation_template, review_template, score_review  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gold = commands.add_parser("prepare-gold")
    gold.add_argument("--dataset", type=Path, default=ROOT / "evals/course_qa.jsonl")
    gold.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("prepare-review")
    review.add_argument("--report", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--review", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose another path to preserve previous work")
        if args.command == "prepare-gold":
            cases = annotation_template(load_cases(args.dataset, None))
            content = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
        elif args.command == "prepare-review":
            content = json.dumps(review_template(args.report), ensure_ascii=False, indent=2) + "\n"
        else:
            review = json.loads(args.review.read_text(encoding="utf-8"))
            content = (
                json.dumps(score_review(args.report, review), ensure_ascii=False, indent=2) + "\n"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(content)
        print(json.dumps({"output": str(args.output), "command": args.command}))
        return 0
    except (ValueError, EvaluationError) as error:
        print(f"Review error: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("Review error: could not read input or create output", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
