"""Backward-compatible entry points around the five-tier Gemma stack.

The previous incarnation of this file was a single monolithic prompt that
tried to be diagnostician, planner, adjudicator, and archivist at once. That
call has been retired in favour of `parallax.tiers.run_stack`, which is what
`app.py` now uses.

What remains here is a thin compatibility layer plus the shared HTTP helpers
used by tiers.py.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request

from parallax.config import (
    API_KEY, CACHED_OUTPUTS_DIR, GEMMA_ENDPOINT, MODEL_NAME, USE_LIVE_GEMMA,
)
from parallax.models import ParallaxAnalysis


try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()


last_live_error: str | None = None


def live_status() -> str | None:
    """The most recent live-call error, or None if the last call succeeded."""
    return last_live_error


def note_live_error(exc: BaseException | str | None) -> None:
    """Public setter so the tier stack can report into the same channel the UI reads."""
    global last_live_error
    if exc is None:
        last_live_error = None
    elif isinstance(exc, str):
        last_live_error = exc
    else:
        last_live_error = f"{type(exc).__name__}: {exc}"


def _call_gemma_api(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(GEMMA_ENDPOINT, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CONTEXT) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _parse_response(text: str) -> ParallaxAnalysis:
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    data = json.loads(text)
    return ParallaxAnalysis.model_validate(data)


def load_cached_result(scenario_id: str) -> ParallaxAnalysis:
    path = os.path.join(CACHED_OUTPUTS_DIR, f"{scenario_id}.json")
    with open(path) as f:
        data = json.load(f)
    return ParallaxAnalysis.model_validate(data)


def run_parallax_analysis(
    scenario: dict,
    retrieved_context: list[dict],
) -> tuple[ParallaxAnalysis, bool]:
    """Legacy single-call analysis retained for the standalone /scenarios path.

    The new demo path is `parallax.tiers.run_stack`. This function is only
    used by any external caller still expecting the original ParallaxAnalysis
    shape and has been kept intentionally simple.
    """
    global last_live_error
    scenario_id = scenario.get("scenario_id", "unknown")

    if USE_LIVE_GEMMA and GEMMA_ENDPOINT:
        prompt = _build_legacy_prompt(scenario, retrieved_context)
        try:
            raw = _call_gemma_api(prompt)
            analysis = _parse_response(raw)
            last_live_error = None
            return analysis, True
        except Exception as exc:
            last_live_error = f"{type(exc).__name__}: {exc}"

    analysis = load_cached_result(scenario_id)
    return analysis, False


def _build_legacy_prompt(scenario: dict, retrieved_context: list[dict]) -> str:
    evidence_list = "\n".join(
        f"- {e['id']} ({e['size_bytes']} bytes): {e['description']}"
        for e in scenario.get("available_evidence", [])
    )
    context_text = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in retrieved_context
    ) or "No additional context retrieved."
    return (
        "You are PARALLAX, an evidence-preservation assistant. "
        "Preserve competing hypotheses and prioritise records for downlink.\n\n"
        f"SCENARIO:\n{json.dumps(scenario, indent=2)}\n\n"
        f"RETRIEVED KNOWLEDGE:\n{context_text}\n\n"
        f"AVAILABLE EVIDENCE:\n{evidence_list}\n\n"
        "Return JSON with keys: event_summary, resolution_status, hypotheses, "
        "preservation_priorities, recommended_follow_up, compression_warning, "
        "uncertainty_statement, source_references."
    )
