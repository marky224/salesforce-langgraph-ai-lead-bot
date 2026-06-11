"""
Shared configuration for the eval suite — imported by both the pytest gate
(``conftest.vcr_config``) and the ``run.py`` CLI so record and replay use the
*exact* same VCR settings (any drift would break body-matching).
"""

from __future__ import annotations

from pathlib import Path

# The model the committed gate cassettes were recorded against. The reasoning ->
# non-reasoning model A/B (PR B's first application, 2026-06-11) picked grok-4.3: it
# held the reasoning baseline's extraction/routing accuracy while cutting pre-stream
# latency, so the cassettes were re-recorded against it and prod's LLM_MODEL moved to
# match. Re-record (one-time, reviewed YAML diff) + bump this constant on any future
# swap. The recorded temperature mirrors the prod default.
RECORD_MODEL = "grok-4.3"
RECORD_TEMPERATURE = 0.7

_HERE = Path(__file__).resolve().parent
DATASET_DIR = _HERE / "datasets"
CASSETTE_DIR = _HERE / "cassettes"

def _scrub_response(response: dict) -> dict:
    """Drop server ``set-cookie`` headers so no session token lands in the YAML."""
    headers = response.get("headers")
    if isinstance(headers, dict):
        for key in list(headers):
            if key.lower() == "set-cookie":
                headers.pop(key, None)
    return response


# VCR config shared by the gate and the CLI.
#
# - ``filter_headers`` strips the bearer token / API keys / cookie from the request,
#   and ``before_record_response`` drops server set-cookie, so no secret is written
#   to the committed YAML (gitleaks scans this repo).
# - ``match_on`` includes ``body``: the entire transcript is baked into the system
#   prompt (extraction/router invoke the LLM with messages=[]), so the request body
#   is deterministic per dataset row. Matching on it makes the gate fail closed —
#   a reworded prompt anchor changes the body, misses the cassette, and errors,
#   forcing a deliberate re-record instead of silently replaying a stale response.
# - ``decode_compressed_response`` stores the response as readable JSON, not gzip,
#   so the cassette diff is a reviewable "the model's behaviour changed" artifact.
VCR_RECORD_CONFIG: dict = {
    "filter_headers": ["authorization", "x-api-key", "api-key", "openai-organization", "cookie"],
    "before_record_response": _scrub_response,
    "decode_compressed_response": True,
    "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
}
