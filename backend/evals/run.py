"""
Eval runner CLI — local / nightly reporting on top of the same datasets,
targets, and evaluators the pytest gate uses.

Two modes:

- ``--mode recorded`` (default): replays the committed cassettes offline. Same
  thing the CI gate does, but prints a readable scorecard instead of asserting.
- ``--mode live``: hits the real provider (needs a real key). This is how the
  reasoning -> non-reasoning **model A/B** is run — point ``--model`` at each
  candidate and compare:

      python -m evals.run --mode live --model grok-4.20-0309-reasoning      --report md
      python -m evals.run --mode live --model grok-4.20-0309-non-reasoning  --report md
      python -m evals.run --mode live --model grok-4.3                       --report md

Run from ``backend/``:  ``python -m evals.run --dimension all --report text``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Make ``app`` / ``evals`` importable regardless of how this is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DIMENSIONS = ("extraction", "routing")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline/online eval suite.")
    parser.add_argument("--dimension", choices=(*DIMENSIONS, "all"), default="all")
    parser.add_argument("--mode", choices=("recorded", "live"), default="recorded")
    parser.add_argument(
        "--model", default=None,
        help="Model override (live mode); defaults to the recorded model. Drives the A/B.",
    )
    parser.add_argument("--report", choices=("text", "json", "md"), default="text")
    parser.add_argument("--output", default=None, help="Write the report to this path instead of stdout.")
    return parser.parse_args()


async def _run_extraction(llm) -> dict:
    from evals.evaluators import aggregate_field_metrics, field_accuracy, parse_success
    from evals.loader import load_dataset
    from evals.targets import run_extraction

    per_row, details = [], []
    for row in load_dataset("extraction.jsonl"):
        predicted = await run_extraction(row, llm)
        accuracy = field_accuracy(predicted, row["expected"])
        per_row.append(accuracy)
        details.append({
            "id": row["id"],
            "scalar": accuracy["scalar"],
            "wrong_or_missing": accuracy["wrong_or_missing"],
            "hallucinated": accuracy["hallucinated"],
            "parse_success": parse_success(predicted, row["expected"]),
        })
    return {"metrics": aggregate_field_metrics(per_row), "details": details}


async def _run_routing(llm) -> dict:
    from evals.evaluators import confusion_matrix, routing_accuracy, routing_match
    from evals.loader import load_dataset
    from evals.targets import run_routing

    results, details = [], []
    for row in load_dataset("routing.jsonl"):
        predicted = await run_routing(row, llm)
        match = routing_match(predicted, row["acceptable_next"])
        results.append(match)
        details.append({"id": row["id"], "predicted": predicted, **match})
    return {"metrics": routing_accuracy(results), "confusion": confusion_matrix(results), "details": details}


async def _run(args: argparse.Namespace, model: str) -> dict:
    from evals.targets import build_eval_llm

    llm = build_eval_llm(model=model)
    runners = {"extraction": _run_extraction, "routing": _run_routing}
    dims = DIMENSIONS if args.dimension == "all" else (args.dimension,)

    report: dict = {"mode": args.mode, "model": model, "dimensions": {}}
    for dim in dims:
        if args.mode == "recorded":
            import vcr

            from evals._config import CASSETTE_DIR, VCR_RECORD_CONFIG

            with vcr.use_cassette(
                str(CASSETTE_DIR / f"{dim}.yaml"), record_mode="none", **VCR_RECORD_CONFIG
            ):
                report["dimensions"][dim] = await runners[dim](llm)
        else:
            report["dimensions"][dim] = await runners[dim](llm)
    return report


def _format_text(report: dict) -> str:
    lines = [f"eval report — mode={report['mode']} model={report['model']}", ""]
    for dim, data in report["dimensions"].items():
        lines.append(f"[{dim}] {data['metrics']}")
        if dim == "routing" and data["metrics"].get("misses"):
            lines.append(f"  misses: {data['metrics']['misses']}")
        if dim == "extraction":
            bad = [d for d in data["details"] if d["wrong_or_missing"] or d["hallucinated"] or d["parse_success"] is False]
            for d in bad:
                lines.append(f"  {d['id']}: missing={d['wrong_or_missing']} hallucinated={d['hallucinated']}")
    return "\n".join(lines)


def _format_md(report: dict) -> str:
    lines = [f"## Eval report — `{report['model']}` ({report['mode']})", ""]
    for dim, data in report["dimensions"].items():
        metrics = data["metrics"]
        lines.append(f"### {dim}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, value in metrics.items():
            if key != "misses":
                lines.append(f"| {key} | {value} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    os.environ.setdefault("LLM_PROVIDER", "xai")
    os.environ["LLM_STRUCTURED_OUTPUT"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if args.mode == "recorded":
        # Replay needs no real secret (VCR intercepts every call).
        os.environ.setdefault("XAI_API_KEY", "xai-dummy-for-replay")

    from evals._config import RECORD_MODEL

    model = args.model or RECORD_MODEL
    report = asyncio.run(_run(args, model))

    if args.report == "json":
        rendered = json.dumps(report, indent=2)
    elif args.report == "md":
        rendered = _format_md(report)
    else:
        rendered = _format_text(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered + "\n")
    else:
        print(rendered)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
