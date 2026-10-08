"""Compare paired API runs while requiring identical source, models and prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrlearnium.eval_tools import compare_reports  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_reports(args.reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {"output": str(args.output), "status_disagreements": report["status_disagreements"]}
            )
        )
        return 0
    except ValueError as error:
        print(f"Comparison error: {error}", file=sys.stderr)
        return 2
    except OSError:
        print(
            "Comparison error: could not read reports or create output (choose a new output path)",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
