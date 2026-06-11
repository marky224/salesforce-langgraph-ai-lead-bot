"""
Opt-in evaluation suite for the AI Sales Lead Bot.

This package is deliberately **separate** from the hermetic ``backend/tests``
suite and from the production Docker image:

- ``backend/tests`` injects fake chat models (``ContentBasedMockLLM``) and never
  touches the network. It verifies *wiring*, not model *behaviour*.
- ``backend/evals`` exercises the **real prompts** against **recorded real-model
  responses** (VCR cassettes), so it measures whether extraction/routing actually
  do the right thing — and fails closed when a prompt-contract regression changes
  the model's request (the e2e mock silently mis-routes on the same reword).

It is never collected by ``pytest tests/`` (different path), never installed into
the image (the Dockerfile copies only ``app/``), and its deps live in
``backend/requirements-eval.txt`` — not ``requirements.txt`` / ``requirements-dev.txt``.

See ``README.md`` for how to run, add cases, and refresh cassettes.
"""
