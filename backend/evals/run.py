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

The ``sim`` dimension (``--dimension sim``, **live only**) drives the real graph
against the personas and prints a per-persona scorecard (see ``simulate.py`` /
``scorecard.py``).

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
    parser.add_argument("--dimension", choices=(*DIMENSIONS, "sim", "all"), default="all")
    parser.add_argument("--mode", choices=("recorded", "live"), default="recorded")
    parser.add_argument(
        "--model", default=None,
        help="Model override (live mode); defaults to the recorded model. Drives the A/B.",
    )
    parser.add_argument("--report", choices=("text", "json", "md"), default="text")
    parser.add_argument("--output", default=None, help="Write the report to this path instead of stdout.")
    parser.add_argument(
        "--judges", action="store_true",
        help="Run the calibrated LLM judges over each sim conversation (--dimension sim only; "
             "extra live cost, needs ANTHROPIC_API_KEY). Persona adherence is validated; summary "
             "faithfulness is advisory (below the kappa bar).",
    )
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


def _transcript_text(trajectory) -> str:
    """Render a trajectory's turns as ``Speaker: text`` lines for a judge's context."""
    return "\n".join(f"{who}: {text}" for who, text in trajectory.turns)


def _judge_score(result):
    return result["score"] if isinstance(result, dict) else result.score


def _judge_comment(result) -> str:
    comment = result.get("comment") if isinstance(result, dict) else getattr(result, "comment", None)
    return (comment or "")[:200]  # keep the rendered scorecard compact


async def _safe_judge(call):
    """
    Run one judge call with a single retry; never raise (judges are advisory).

    openevals can miss the structured ``score`` on a nondeterministic parse hiccup; a
    retry usually clears it. On repeated failure return ``{"error": msg}`` so one flaky
    judge degrades to an error row instead of crashing the whole sim run.
    """
    last: Exception | None = None
    for _ in range(2):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 — advisory feature must not crash the run
            last = exc
    return {"error": f"{type(last).__name__}: {last}"[:200]}


def _judge_row(key: str, result) -> dict:
    from evals.calibration import is_validated

    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"], "validated": is_validated(key)}
    return {
        "score": _judge_score(result),
        "validated": is_validated(key),
        "comment": _judge_comment(result),
    }


async def _judge_trajectory(trajectory, persona_ev, faith_ev) -> dict:
    """
    Run the calibrated judges over one finished conversation (live, advisory rows).

    Persona adherence grades TARS's turns against the transcript; summary faithfulness
    grades the Salesforce ``transcript_summary`` against the transcript, and only when a
    summary exists (i.e. the run reached COMPLETE). Each row carries ``validated`` from the
    recorded kappa; a judge call that fails twice degrades to an error row (``_safe_judge``).
    """
    transcript = _transcript_text(trajectory)
    judges: dict = {}

    persona_res = await _safe_judge(
        lambda: persona_ev(inputs=transcript, outputs="\n".join(trajectory.tars_turns))
    )
    judges["persona_adherence"] = _judge_row("persona_adherence", persona_res)

    summary = (trajectory.final_state or {}).get("transcript_summary")
    if summary:
        faith_res = await _safe_judge(lambda: faith_ev(outputs=summary, context=transcript))
        judges["summary_faithfulness"] = _judge_row("summary_faithfulness", faith_res)
    return judges


async def _run_sim(model: str, judges: bool = False) -> dict:
    """
    Drive the real graph against each persona (live only) → per-persona scorecard.

    The system-under-test runs ``model`` (the ``--model`` override); the simulated
    visitor is held fixed at the recorded grok-4.3 driver (decision #4) so a SUT swap
    stays comparable. Salesforce is mocked inside ``simulate`` — zero CRM writes. With
    ``judges=True`` the calibrated judges (claude-sonnet-4-6) also grade each finished
    conversation, attaching advisory rows (extra live cost, needs ANTHROPIC_API_KEY).
    """
    from app.graph.graph import build_graph
    from app.graph.nodes import set_llm
    from evals.loader import load_dataset
    from evals.scorecard import score_persona
    from evals.simulate import simulate
    from evals.targets import build_eval_llm

    set_llm(build_eval_llm(model=model))
    graph = build_graph()
    user_llm = build_eval_llm()  # fixed visitor driver (RECORD_MODEL)

    persona_ev = faith_ev = None
    if judges:
        from evals.judges import persona_adherence_judge, summary_faithfulness_judge
        persona_ev, faith_ev = persona_adherence_judge(), summary_faithfulness_judge()

    rows = []
    for persona in load_dataset("personas.jsonl"):
        traj = await simulate(
            graph, persona, user_llm, max_turns=persona["expected"].get("max_turns", 12)
        )
        row = score_persona(traj, persona)
        if judges:
            row["judges"] = await _judge_trajectory(traj, persona_ev, faith_ev)
        rows.append(row)
    return {"mode": "live", "model": model, "sim": rows, "judges": judges}


async def _run(args: argparse.Namespace, model: str) -> dict:
    if args.dimension == "sim":
        return await _run_sim(model, judges=args.judges)

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
    if "sim" in report:
        return _format_sim_text(report)
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
    if "sim" in report:
        return _format_sim_md(report)
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


def _format_sim_text(report: dict) -> str:
    lines = [f"sim scorecard — model={report['model']} (live, report-only)", ""]
    for row in report["sim"]:
        verdict = "PASS" if row["overall_pass"] else "FAIL"
        lines.append(
            f"[{verdict}] {row['id']}: score={row['score']['value']} band={row['score']['band']} "
            f"complete={row['reached_complete']['value']} turns={row['turns']['taken']}"
        )
        if row["must_capture"]["missing"]:
            lines.append(f"    missing: {row['must_capture']['missing']}")
        if not row["no_stage_loop"]:
            lines.append("    stage loop detected")
        if not row["reask"]["ok"]:
            fields = [r["field"] for r in row["reask"]["reasks"]]
            lines.append(f"    re-asked already-captured fields: {fields}")
        voice = row["voice"]
        if not (voice["markdown_clean"] and voice["brevity_ok"]):
            lines.append(
                f"    voice: markdown_clean={voice['markdown_clean']} "
                f"brevity_ok={voice['brevity_ok']} max_sentences={voice['max_sentences_seen']}"
            )
        for key, j in (row.get("judges") or {}).items():
            if "error" in j:
                lines.append(f"    judge[{key}]: ERROR {j['error']}")
            else:
                tag = "validated" if j["validated"] else "advisory"
                lines.append(f"    judge[{key}]: score={j['score']} ({tag})")
    return "\n".join(lines)


def _judges_cell(judges: dict | None) -> str:
    """Compact one-cell judge summary for the md table (``—`` when judges were off)."""
    if not judges:
        return "—"
    short = {"persona_adherence": "persona", "summary_faithfulness": "faith"}
    parts = []
    for key, j in judges.items():
        name = short.get(key, key)
        if "error" in j:
            parts.append(f"{name}=ERR")
        else:
            parts.append(f"{name}={j['score']}" + ("✓" if j["validated"] else " (adv)"))
    return " ".join(parts)


def _format_sim_md(report: dict) -> str:
    show_judges = bool(report.get("judges"))
    header = "| persona | pass | score | band | complete | turns | must-capture | stage-loop | reask | voice |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if show_judges:
        header += " judges |"
        sep += "---|"
    lines = [
        f"## Sim scorecard — `{report['model']}` (live, report-only)",
        "",
        header,
        sep,
    ]
    for row in report["sim"]:
        v = row["voice"]
        voice = "ok" if (v["markdown_clean"] and v["brevity_ok"]) else f"md={v['markdown_clean']},brev={v['brevity_ok']}"
        cap = "ok" if row["must_capture"]["ok"] else ", ".join(row["must_capture"]["missing"])
        reask = "ok" if row["reask"]["ok"] else "REASK"
        cells = (
            f"| {row['id']} | {'✅' if row['overall_pass'] else '❌'} | {row['score']['value']} "
            f"| {row['score']['band']} | {row['reached_complete']['value']} | {row['turns']['taken']} "
            f"| {cap} | {'ok' if row['no_stage_loop'] else 'LOOP'} | {reask} | {voice} |"
        )
        if show_judges:
            cells += f" {_judges_cell(row.get('judges'))} |"
        lines.append(cells)
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    if args.dimension == "sim" and args.mode != "live":
        print("error: --dimension sim requires --mode live (no cassette to replay)", file=sys.stderr)
        return 2

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
