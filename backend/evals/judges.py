"""
Calibrated LLM-as-judge evaluators (PR D) — the *quality* layer the deterministic
scorecard cannot measure.

Two judges, both **eval_live only** (real frontier-model calls; never the offline
gate, never the hermetic suite, never the image):

- ``persona_adherence`` — does TARS hold its deadpan-but-warm persona across a whole
  conversation?  Hand-rolled prompt grounding the rubric in the real
  ``app.graph.prompts.PERSONA``.  openevals ships a generic ``AGENT_TONE_PROMPT``,
  but its rubric scores "robotic / cold / impersonal" tone as *inappropriate* —
  which is exactly TARS's intended register — so a generic tone judge would
  systematically misgrade this persona.  A persona-grounded prompt is required for
  correctness, not merely preferred.
- ``summary_faithfulness`` — is the Salesforce ``transcript_summary`` grounded in the
  actual chat, or does it hallucinate?  Reuses openevals' prebuilt
  ``RAG_GROUNDEDNESS_PROMPT`` (generic-correct: claim vs. context) with the summary
  as the claim (``outputs``) and the full transcript as the context (``context``).

The judge model is a **frontier Claude** (``claude-sonnet-4-6``), deliberately a
*different* family from the grok-4.3 system-under-test: a Grok-judging-Grok setup
risks correlated blind spots (the judge sharing the SUT's failure modes).  It is
built through the app's own ``get_llm`` (Anthropic provider) so the key is sourced
from ``.env`` via Settings and the per-call timeout / retry knobs apply — the same
path ``targets.build_eval_llm`` uses for the xAI SUT.  Requires ``langchain-anthropic``
(an eval-only dependency) + ``ANTHROPIC_API_KEY``.

These are **async** evaluators (``create_async_llm_as_judge``) so they compose with
the async sim loop; each returns an openevals ``EvaluatorResult``
(``{"key", "score", "comment", ...}``).  Calibration against human labels (Cohen's κ)
and wiring into the scorecard are PR D's later commits — a judge counts toward the
scorecard only once it clears the documented κ bar.
"""

from __future__ import annotations

from typing import Any

from openevals.llm import create_async_llm_as_judge
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT

from app.config import LLMProvider, get_llm
from app.graph.prompts import PERSONA

# Frontier judge — a different model family from the grok-4.3 SUT to avoid correlated
# blind spots. claude-sonnet-4-6 was chosen for calibration/cost; claude-opus-4-8 is
# the swap for maximum calibration.
JUDGE_MODEL = "claude-sonnet-4-6"


def build_judge_llm(model: str = JUDGE_MODEL) -> Any:
    """
    Build the Anthropic judge model via the app's own LLM factory.

    Reuses ``app.config.get_llm`` (Anthropic provider) so the key is sourced from
    ``.env`` through Settings and the per-call timeout / retry knobs apply — mirroring
    how ``targets.build_eval_llm`` builds the xAI system-under-test.  ``temperature=0``
    for stable, repeatable verdicts (Sonnet 4.6 accepts it).  Raises if
    ``ANTHROPIC_API_KEY`` is unset, so it is never reached on the offline gate — the
    plumbing tests pass an explicit fake judge instead of calling this.
    """
    judge = get_llm(provider=LLMProvider.ANTHROPIC, model=model, temperature=0.0)
    # The judge reasons over a full transcript before emitting its score; the app default
    # max_tokens=1024 can truncate that reasoning mid-structured-output, leaving openevals
    # without a ``score`` (KeyError on a long conversation). Give the judge room — this
    # touches only the judge model, not the app's conversational nodes.
    judge.max_tokens = 4096
    return judge


# PERSONA is plain prose with no ``{`` / ``}`` (verified), so it embeds safely in an
# openevals ``str.format`` template alongside the ``{inputs}`` / ``{outputs}``
# placeholders openevals fills at call time.
PERSONA_ADHERENCE_PROMPT = (
    "You are an expert conversation evaluator scoring whether an AI sales advisor "
    "named TARS stayed true to its intended persona across a chat.\n\n"
    "TARS is SUPPOSED to sound like the persona below — judge against THIS rubric, "
    "not a generic 'friendly assistant' standard. Dry, deadpan, sparing wit is "
    "ON-persona here, not a defect:\n\n"
    "<persona>\n" + PERSONA + "\n</persona>\n\n"
    "The full conversation for context (visitor and TARS):\n"
    "<conversation>\n{inputs}\n</conversation>\n\n"
    "Score ONLY TARS's own messages:\n"
    "<tars_messages>\n{outputs}\n</tars_messages>\n\n"
    "Scoring — choose exactly one:\n"
    "- 1.0 — on-voice: dry/deadpan wit used sparingly and well, concise (about 2-4 "
    "sentences), warm underneath, genuinely helpful, NEVER sarcastic at the visitor "
    "or their company/pain, drops the humor when they share a real frustration, no "
    "markdown.\n"
    "- 0.5 — slightly off: mostly in character but with lapses — too verbose, a flat "
    "or generic reply, a joke that lands awkwardly, or mild over-eagerness.\n"
    "- 0.0 — off-persona: sarcastic AT the visitor, pushy like a telemarketer, "
    "robotic-generic with no personality, or a wall of text.\n"
)


def persona_adherence_judge(judge: Any | None = None) -> Any:
    """
    Async judge: did TARS hold its persona?  Score in {0.0, 0.5, 1.0} + reasoning.

    Call as ``await ev(inputs=<full transcript>, outputs=<TARS turns>)`` → an openevals
    ``EvaluatorResult`` (``{"key": "persona_adherence", "score", "comment", ...}``).
    Pass an explicit ``judge`` (a LangChain chat model) to avoid building the real
    Anthropic client — the offline plumbing test does this with a fake model, so it
    never touches the network or needs a key.
    """
    return create_async_llm_as_judge(
        prompt=PERSONA_ADHERENCE_PROMPT,
        judge=judge or build_judge_llm(),
        feedback_key="persona_adherence",
        choices=[0.0, 0.5, 1.0],
        use_reasoning=True,
    )


def summary_faithfulness_judge(judge: Any | None = None) -> Any:
    """
    Async judge: is the Salesforce ``transcript_summary`` grounded in the chat?

    Reuses openevals' prebuilt ``RAG_GROUNDEDNESS_PROMPT`` (boolean grounded /
    not-grounded + reasoning).  Call as
    ``await ev(outputs=<transcript_summary>, context=<full transcript>)`` —
    groundedness ignores ``inputs`` by design, comparing the claim (the summary) only
    against the context (the transcript).
    """
    return create_async_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        judge=judge or build_judge_llm(),
        feedback_key="summary_faithfulness",
        use_reasoning=True,
    )
