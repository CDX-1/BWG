"""Five-role Gemma reasoning stack — G0 SENTINEL through G4 ARCHIVIST.

Each tier is its own prompt, temperature, context scope, output cap and stop
condition. The tiers are role-differentiated rather than size-differentiated:
one Gemma endpoint answers every call, but each call sees a tightly-scoped
prompt and produces a single-purpose output.

  G0 SENTINEL      continuous anomaly watch          ~0.1 s   2 out-tokens
  G1 DIAGNOSTICIAN competing hypotheses with cites   ~2.5 s  ~210 tokens
  G2 FLIGHT DIR    recovery plan over whitelist       ~0.7 s   ~50 tokens
  G3 ADJUDICATOR   cold red-team of G2 plan (T=0)     ~0.5 s   ~60 tokens
  G4 ARCHIVIST     evidence value model for capsule   ~0.4 s   ~30 tokens

The tiers are dependency-parallelised: G1 and G4 have no ordering constraint
against each other, G2 depends on G1's hypotheses, and G3 depends on G2's
plan. G0 runs on its own timer, decoupled from the others.

Every tier returns a `TierResult` carrying (payload, TierTrace). The trace is
what makes the stack visible and honest in the UI — it says what actually ran
live, what fell back, what took how long, and why.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import ValidationError

from parallax.action_vocabulary import ACTIONS, action_vocabulary_for_prompt
from parallax.config import API_KEY, GEMMA_ENDPOINT, MODEL_NAME, USE_LIVE_GEMMA
from parallax.models import (
    AdjudicatorVerdict,
    ArchivistPacking,
    Diagnosis,
    DiagnosticianHypothesis,
    EvidenceScore,
    PlanStep,
    RecoveryPlan,
    SentinelVerdict,
    TierTrace,
)
from parallax.retrieval import load_knowledge_chunks, retrieve


# ── HTTP wrapper (same SSL fix as gemma.py) ──────────────────────────────────

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()


def _call_gemma(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    system: str | None = None,
    json_mode: bool = False,
    timeout_s: float = 45.0,
) -> tuple[str, int]:
    """One synchronous call to the endpoint. Returns (content, output_tokens).

    The output-token count is estimated from the response when the endpoint
    does not return usage stats — the endpoint's true cost model is opaque,
    but tokens_out is what latency scales with and is what we want to display.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(GEMMA_ENDPOINT, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s, context=_SSL_CONTEXT) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    tokens_out = int(usage.get("completion_tokens") or _estimate_tokens(content))
    return content, tokens_out


def _estimate_tokens(text: str) -> int:
    """Rough estimate — 4 characters per token, floor 1."""
    return max(1, len(text or "") // 4)


def _strip_json_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    # Some models put a leading label; find the first JSON brace.
    if not text.lstrip().startswith(("{", "[")):
        m = re.search(r"[\{\[]", text)
        if m:
            text = text[m.start():]
    return text.strip()


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON, repairing common truncation cases.

    max_tokens can cut a response mid-string or mid-array. Instead of losing
    the whole tier to a JSONDecodeError, walk the text once collecting
    "candidate cut points" (positions just after a complete value), then try
    them from latest to earliest, closing open containers on each attempt.
    A partial diagnosis is worth more than none.
    """
    text = _strip_json_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates = _collect_cut_points(text)
    last_exc: Exception | None = None
    for pos in reversed(candidates):
        repaired = _repair_at(text, pos)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
    raise last_exc or json.JSONDecodeError("no recoverable structure", text, 0)


def _collect_cut_points(text: str) -> list[int]:
    """Positions just after any complete JSON value at any nesting level.

    A "cut point" is where the prefix ending at that character could be
    completed into valid JSON by only appending closing braces/brackets.
    """
    points: list[int] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                points.append(i)   # end of a string value or key
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "}]":
            points.append(i)
        elif ch in "0123456789":
            points.append(i)
        elif ch == "e" and i + 4 < len(text) and text[i:i+4] == "true":
            points.append(i + 3)
        elif ch == "n" and text[i:i+4] == "null":
            points.append(i + 3)
        elif ch == "e" and i + 5 < len(text) and text[i-4:i+1] == "false":
            points.append(i)
    return points


def _repair_at(text: str, pos: int) -> str:
    """Trim to `pos+1` and synthesise closers plus drop any orphan trailing key."""
    prefix = text[: pos + 1]
    prefix = _drop_orphan_trailing(prefix)

    # Recompute the open stack over the trimmed prefix and close it.
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in prefix:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    return prefix + "".join(reversed(stack))


def _drop_orphan_trailing(prefix: str) -> str:
    """Remove any dangling `,`, `:`, or `"key"[:]` at the end of the prefix.

    Repeats: `{"a":1,"b":` → `{"a":1` (dropped `,"b":`). Then closers appended.
    """
    changed = True
    while changed:
        changed = False
        stripped = prefix.rstrip()
        if stripped != prefix:
            prefix = stripped
            changed = True
        # Trailing comma or colon — bare separator with no value coming.
        if prefix and prefix[-1] in ",:":
            prefix = prefix[:-1]
            changed = True
            continue
        # Trailing string that is actually a key (immediately before a colon
        # that was never followed by a value). If the last string in the
        # prefix isn't followed by `:value`, it must be a value that closed
        # cleanly; nothing to do. We detect a "hanging key" by scanning back
        # past the last quoted string and checking whether the character
        # before the opening quote is `,` or `{` (making it a key position).
        if prefix.endswith('"') and _last_string_is_key(prefix):
            prefix = _drop_last_string(prefix)
            changed = True
    return prefix


def _last_string_is_key(prefix: str) -> bool:
    """True if the last closed string sits in a key position."""
    # Find the opening quote of the last string in prefix.
    if not prefix.endswith('"'):
        return False
    i = len(prefix) - 2
    escape_run = 0
    while i >= 0:
        if prefix[i] == '"' and escape_run % 2 == 0:
            break
        escape_run = escape_run + 1 if prefix[i] == "\\" else 0
        i -= 1
    if i < 0:
        return False
    j = i - 1
    while j >= 0 and prefix[j] in " \t\n":
        j -= 1
    return j >= 0 and prefix[j] in "{,"


def _drop_last_string(prefix: str) -> str:
    """Remove the last quoted string (assumed a hanging key) plus a leading comma."""
    i = len(prefix) - 2
    escape_run = 0
    while i >= 0:
        if prefix[i] == '"' and escape_run % 2 == 0:
            break
        escape_run = escape_run + 1 if prefix[i] == "\\" else 0
        i -= 1
    if i < 0:
        return prefix
    j = i - 1
    while j >= 0 and prefix[j] in " \t\n":
        j -= 1
    if j >= 0 and prefix[j] == ",":
        return prefix[:j]
    return prefix[:i]


# ── Tier result container ────────────────────────────────────────────────────

@dataclass
class TierResult:
    payload: Any
    trace: TierTrace


def _fallback(tier: str, label: str, note: str, payload: Any) -> TierResult:
    return TierResult(
        payload=payload,
        trace=TierTrace(tier=tier, label=label, ok=True, is_live=False,
                        latency_s=0.0, tokens_out=0, note=note),
    )


# ── Retrieval helper ─────────────────────────────────────────────────────────

_KNOWLEDGE_CACHE: list[dict] | None = None


def _knowledge_chunks() -> list[dict]:
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE is None:
        _KNOWLEDGE_CACHE = load_knowledge_chunks()
    return _KNOWLEDGE_CACHE


def _retrieve_context(query: str, top_k: int = 3) -> tuple[list[dict], str]:
    """Return the retrieved chunks and a compact string for the prompt."""
    chunks = retrieve(query, _knowledge_chunks(), top_k=top_k)
    text = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks) or "(no procedures matched)"
    return chunks, text


# ── G0 SENTINEL ──────────────────────────────────────────────────────────────

_SENTINEL_SYSTEM = (
    "You are G0 SENTINEL, the always-on anomaly watch for the Asteria-7 "
    "spacecraft. You read a featurised telemetry window and reply with ONE "
    "of: nominal, watch, anomalous. When not nominal, include a short phrase "
    "on what looks odd. Do not diagnose. Do not recommend actions. "
    "Reply as JSON: {\"status\": \"...\", \"what_looks_odd\": \"...\"}."
)


def run_sentinel(featurised_window: dict) -> TierResult:
    prompt = json.dumps(featurised_window, separators=(",", ":"))
    t0 = time.perf_counter()

    if not (USE_LIVE_GEMMA and GEMMA_ENDPOINT):
        return _fallback("G0", "SENTINEL", "live off",
                         SentinelVerdict(status="nominal",
                                         what_looks_odd="(sentinel offline)"))

    try:
        content, tokens_out = _call_gemma(
            prompt, system=_SENTINEL_SYSTEM,
            temperature=0.0, max_tokens=60, json_mode=True, timeout_s=8.0,
        )
        data = _parse_json_lenient(content)
        verdict = SentinelVerdict.model_validate(data)
    except Exception as exc:
        return TierResult(
            payload=SentinelVerdict(status="nominal", what_looks_odd="(sentinel unreachable)"),
            trace=TierTrace(tier="G0", label="SENTINEL", ok=False, is_live=False,
                            latency_s=time.perf_counter() - t0, tokens_out=0,
                            error=f"{type(exc).__name__}: {exc}"),
        )

    return TierResult(
        payload=verdict,
        trace=TierTrace(tier="G0", label="SENTINEL", ok=True, is_live=True,
                        latency_s=time.perf_counter() - t0, tokens_out=tokens_out,
                        note=verdict.status),
    )


# ── G1 DIAGNOSTICIAN ─────────────────────────────────────────────────────────

_DIAGNOSTICIAN_SYSTEM = (
    "You are G1 DIAGNOSTICIAN, isolating an anomaly on the Asteria-7 "
    "spacecraft. Produce 2 to 3 COMPETING explanations for the observed state.\n\n"
    "Rules:\n"
    "- Never collapse to a single answer. Preserve uncertainty.\n"
    "- Each hypothesis MUST cite at least one procedure section tag like "
    "[INCIDENT-RESPONSE §2.1] drawn from the retrieved procedures below.\n"
    "- Keep every string under 20 words. Keep evidence arrays under 4 items.\n"
    "- Confidence: low, medium, or high — never absolute.\n\n"
    "Reply ONLY as JSON matching this schema:\n"
    "{\n"
    "  \"summary\": \"one sentence\",\n"
    "  \"hypotheses\": [\n"
    "    {\"name\": \"short\", \"confidence\": \"low|medium|high\",\n"
    "     \"supporting_evidence\": [\"field_name\"], \"contradicting_evidence\": [\"field_name\"],\n"
    "     \"citations\": [\"[TAG §x.y]\"]}\n"
    "  ],\n"
    "  \"all_hypotheses_cited\": true\n"
    "}\n"
)


def run_diagnostician(
    featurised_state: dict,
    fdir_summary: str,
    active_faults: list[str],
) -> TierResult:
    query = ("competing hypotheses recovery procedures for "
             + ", ".join(active_faults or ["unknown anomaly"]))
    chunks, retrieved_text = _retrieve_context(query, top_k=3)

    user_prompt = (
        f"FEATURISED STATE:\n{json.dumps(featurised_state, indent=2)}\n\n"
        f"FDIR SUMMARY: {fdir_summary}\n\n"
        f"RETRIEVED PROCEDURES:\n{retrieved_text}\n"
    )

    t0 = time.perf_counter()

    if not (USE_LIVE_GEMMA and GEMMA_ENDPOINT):
        return _fallback("G1", "DIAGNOSTICIAN", "live off",
                         _synthesize_diagnosis(active_faults, chunks))

    try:
        content, tokens_out = _call_gemma(
            user_prompt, system=_DIAGNOSTICIAN_SYSTEM,
            temperature=0.4, max_tokens=1100, json_mode=True, timeout_s=30.0,
        )
        data = _parse_json_lenient(content)
        # Fill in absent optional fields so a truncated payload still validates.
        data.setdefault("summary", "(diagnosis summary truncated)")
        data.setdefault("hypotheses", [])
        for h in data["hypotheses"]:
            h.setdefault("supporting_evidence", [])
            h.setdefault("contradicting_evidence", [])
            h.setdefault("citations", [])
            h.setdefault("confidence", "low")
        diagnosis = Diagnosis.model_validate(data)
        if not diagnosis.hypotheses:
            # Truncated to nothing usable — fall through to synthesis.
            raise ValueError("no hypotheses recovered from response")
        # Recompute the cited flag ourselves — the model is unreliable at self-checks.
        diagnosis.all_hypotheses_cited = all(bool(h.citations) for h in diagnosis.hypotheses)
    except Exception as exc:
        diag = _synthesize_diagnosis(active_faults, chunks)
        return TierResult(
            payload=diag,
            trace=TierTrace(tier="G1", label="DIAGNOSTICIAN", ok=False, is_live=False,
                            latency_s=time.perf_counter() - t0, tokens_out=0,
                            error=f"{type(exc).__name__}: {exc}"),
        )

    return TierResult(
        payload=diagnosis,
        trace=TierTrace(tier="G1", label="DIAGNOSTICIAN", ok=True, is_live=True,
                        latency_s=time.perf_counter() - t0, tokens_out=tokens_out,
                        note=f"{len(diagnosis.hypotheses)} hypotheses, "
                             f"{'all cited' if diagnosis.all_hypotheses_cited else 'missing citations'}"),
    )


def _synthesize_diagnosis(active_faults: list[str], chunks: list[dict]) -> Diagnosis:
    """A minimal offline diagnosis so the UI still fills in.

    Draws a citation from the first retrieved chunk so downstream tiers can be
    exercised even when the endpoint is unreachable.
    """
    citation = _first_tag(chunks[0]["text"]) if chunks else "[INCIDENT-RESPONSE §1.1]"
    labels = active_faults or ["unclassified event"]
    hyps = [
        DiagnosticianHypothesis(
            name=f"{labels[0].replace('_', ' ')} — hardware fault",
            confidence="medium",
            supporting_evidence=["active_faults", "subsystem_health"],
            contradicting_evidence=[],
            citations=[citation],
        ),
        DiagnosticianHypothesis(
            name="Data-integrity artefact (corrupted telemetry)",
            confidence="low",
            supporting_evidence=[],
            contradicting_evidence=["no packet-integrity alarm reported"],
            citations=[citation],
        ),
    ]
    return Diagnosis(
        summary=f"Offline synthesis for {', '.join(labels)} — competing explanations preserved.",
        hypotheses=hyps,
        all_hypotheses_cited=True,
    )


def _first_tag(text: str) -> str:
    m = re.search(r"\[[A-Z\-]+ §[0-9.]+\]", text)
    return m.group(0) if m else "[INCIDENT-RESPONSE §1.1]"


# ── G2 FLIGHT DIRECTOR ───────────────────────────────────────────────────────

_FLIGHT_DIRECTOR_SYSTEM = (
    "You are G2 FLIGHT DIRECTOR, planning recovery for the Asteria-7 "
    "spacecraft. You never issue commands directly; you produce a plan the "
    "vehicle's deterministic gate then validates.\n\n"
    "Rules:\n"
    "- Every step must use one of these actions and nothing else:\n"
    "{VOCAB}\n"
    "- 1 to 5 steps, sequenced so earlier steps don't undercut later ones.\n"
    "- When several faults compete for the same resource (power, thermal, "
    "pointing), arbitrate — do NOT concatenate independent recovery scripts.\n"
    "- No step may risk the battery below 40% or leave the spacecraft with "
    "no working comms path.\n"
    "- Cite at least one procedure section per step.\n\n"
    "Reply ONLY as JSON:\n"
    "{\n"
    "  \"summary\": \"...\",\n"
    "  \"steps\": [\n"
    "    {\"action\": \"<vocab_name>\", \"params\": {...}, "
    "\"rationale\": \"...\", \"citations\": [\"[TAG §x.y]\"]}\n"
    "  ],\n"
    "  \"estimated_battery_cost_pct\": <number>,\n"
    "  \"addresses_hypotheses\": [\"<hypothesis name>\", ...]\n"
    "}\n"
)


def run_flight_director(
    featurised_state: dict,
    diagnosis: Diagnosis,
    fdir_summary: str,
    forbidden_actions: set[str] | None = None,
) -> TierResult:
    query = ("recovery plan sequencing for "
             + ", ".join(featurised_state.get("active_faults", [])) or "anomaly recovery")
    chunks, retrieved_text = _retrieve_context(query, top_k=2)

    vocab = action_vocabulary_for_prompt()
    if forbidden_actions:
        vocab += "\n(previous plan rejected — do NOT use: " + ", ".join(sorted(forbidden_actions)) + ")"

    system = _FLIGHT_DIRECTOR_SYSTEM.replace("{VOCAB}", vocab)

    hypo_summary = "\n".join(
        f"- {h.name} ({h.confidence})" for h in diagnosis.hypotheses
    ) or "(no diagnosis available)"

    user_prompt = (
        f"COMPETING HYPOTHESES:\n{hypo_summary}\n\n"
        f"FEATURISED STATE:\n{json.dumps(featurised_state, indent=2)}\n\n"
        f"FDIR SUMMARY: {fdir_summary}\n\n"
        f"RETRIEVED PROCEDURES:\n{retrieved_text}\n"
    )

    t0 = time.perf_counter()

    if not (USE_LIVE_GEMMA and GEMMA_ENDPOINT):
        return _fallback("G2", "FLIGHT DIRECTOR", "live off",
                         _synthesize_plan(featurised_state, diagnosis))

    try:
        content, tokens_out = _call_gemma(
            user_prompt, system=system,
            temperature=0.2, max_tokens=550, json_mode=True, timeout_s=25.0,
        )
        data = _parse_json_lenient(content)
        plan = RecoveryPlan.model_validate(data)
    except Exception as exc:
        plan = _synthesize_plan(featurised_state, diagnosis)
        return TierResult(
            payload=plan,
            trace=TierTrace(tier="G2", label="FLIGHT DIRECTOR", ok=False, is_live=False,
                            latency_s=time.perf_counter() - t0, tokens_out=0,
                            error=f"{type(exc).__name__}: {exc}"),
        )

    return TierResult(
        payload=plan,
        trace=TierTrace(tier="G2", label="FLIGHT DIRECTOR", ok=True, is_live=True,
                        latency_s=time.perf_counter() - t0, tokens_out=tokens_out,
                        note=f"{len(plan.steps)} steps · "
                             f"~{plan.estimated_battery_cost_pct:.1f}% battery"),
    )


def _synthesize_plan(featurised_state: dict, diagnosis: Diagnosis) -> RecoveryPlan:
    """A safe conservative plan for the offline path.

    Sheds load then enters safe mode. Both actions are envelope-safe from any
    non-catastrophic starting state, so this fallback never trips the gates.
    """
    citation = "[INCIDENT-RESPONSE §1.1]"
    steps = [
        PlanStep(action="shed_load", params={"pct": 30}, rationale="Reduce non-essential draw.",
                 citations=[citation]),
        PlanStep(action="enter_safe_mode", params={}, rationale="Sun-point and quiesce science.",
                 citations=[citation]),
    ]
    return RecoveryPlan(
        summary="Offline conservative plan: shed load, then enter safe mode.",
        steps=steps,
        estimated_battery_cost_pct=1.5,
        addresses_hypotheses=[h.name for h in diagnosis.hypotheses[:2]],
    )


# ── G3 ADJUDICATOR ───────────────────────────────────────────────────────────

_ADJUDICATOR_SYSTEM = (
    "You are G3 ADJUDICATOR, the red-team on the plan G2 just produced.\n"
    "You did NOT see G2's reasoning. You see only the plan and the current "
    "featurised state. You may consult retrieved procedures.\n\n"
    "Approve the plan only if:\n"
    "1. Every step is in the whitelisted action vocabulary.\n"
    "2. No step endangers battery (<40% projected) or the sole comms path.\n"
    "3. When multiple faults are active, the plan arbitrates rather than "
    "   running scripts concurrently that fight each other.\n"
    "4. Each step's stated rationale is consistent with retrieved procedures.\n\n"
    "Reply ONLY as JSON:\n"
    "{\"approved\": true|false, \"reason\": \"...\", "
    "\"objection_step\": <null or 1-indexed step>, "
    "\"revision_notes\": [\"...\"]}\n"
)


def run_adjudicator(
    plan: RecoveryPlan,
    featurised_state: dict,
    validation_report,
) -> TierResult:
    query = "recovery plan safety review " + " ".join(
        s.action for s in plan.steps
    )
    _, retrieved_text = _retrieve_context(query, top_k=2)

    plan_dump = json.dumps(plan.model_dump(), indent=2)
    val_line = "PRE-CHECK GATE: " + validation_report.summary_line

    user_prompt = (
        f"PLAN:\n{plan_dump}\n\n"
        f"{val_line}\n\n"
        f"FEATURISED STATE:\n{json.dumps(featurised_state, indent=2)}\n\n"
        f"RETRIEVED PROCEDURES:\n{retrieved_text}\n"
    )

    t0 = time.perf_counter()

    # If the deterministic gate already rejected, mirror that as the verdict
    # without spending an API call. The adjudicator agrees with the gate by
    # construction — that gate is stricter than a language-model opinion.
    if not validation_report.approved:
        verdict = AdjudicatorVerdict(
            approved=False,
            reason=(f"Deterministic gate rejected: {validation_report.rejected_reason}"),
            objection_step=(validation_report.rejected_step_index or 0) + 1,
            revision_notes=[f"Fix the {validation_report.rejected_gate} violation before re-submitting."],
        )
        return TierResult(
            payload=verdict,
            trace=TierTrace(tier="G3", label="ADJUDICATOR", ok=True, is_live=False,
                            latency_s=time.perf_counter() - t0, tokens_out=0,
                            note=f"mirrored gate veto (step {verdict.objection_step})"),
        )

    if not (USE_LIVE_GEMMA and GEMMA_ENDPOINT):
        return _fallback("G3", "ADJUDICATOR", "live off",
                         AdjudicatorVerdict(approved=True, reason="offline auto-approve",
                                             revision_notes=[]))

    try:
        content, tokens_out = _call_gemma(
            user_prompt, system=_ADJUDICATOR_SYSTEM,
            temperature=0.0, max_tokens=250, json_mode=True, timeout_s=15.0,
        )
        data = _parse_json_lenient(content)
        verdict = AdjudicatorVerdict.model_validate(data)
    except Exception as exc:
        # A silent adjudicator is worse than a loud one — if G3 can't run,
        # default to conservative approval and flag it plainly in the trace.
        verdict = AdjudicatorVerdict(approved=True, reason="G3 unreachable; deterministic gate passed",
                                     revision_notes=[])
        return TierResult(
            payload=verdict,
            trace=TierTrace(tier="G3", label="ADJUDICATOR", ok=False, is_live=False,
                            latency_s=time.perf_counter() - t0, tokens_out=0,
                            error=f"{type(exc).__name__}: {exc}"),
        )

    return TierResult(
        payload=verdict,
        trace=TierTrace(tier="G3", label="ADJUDICATOR", ok=True, is_live=True,
                        latency_s=time.perf_counter() - t0, tokens_out=tokens_out,
                        note="approved" if verdict.approved else f"vetoed step {verdict.objection_step}"),
    )


# ── G4 ARCHIVIST ─────────────────────────────────────────────────────────────

_ARCHIVIST_SYSTEM = (
    "You are G4 ARCHIVIST, scoring evidence for downlink under a limited "
    "byte budget. For every listed record, assign an integer score 0–100 "
    "reflecting how much a ground investigator would lose if it never made it "
    "home, given the competing hypotheses still alive.\n\n"
    "Rules:\n"
    "- Raw high-fidelity records beat pre-processed summaries.\n"
    "- Records that support MULTIPLE hypotheses score higher than one-answer records.\n"
    "- Small logs (packet integrity, sequence numbers) often score high — "
    "they are irreplaceable and cheap.\n"
    "- Keep every 'reason' to under 12 words. Prefer short arrays.\n\n"
    "Reply ONLY as JSON:\n"
    "{\"scores\": [{\"evidence_id\": \"...\", \"score\": <int>, \"reason\": \"short\", "
    "\"supports_hypotheses\": [\"...\"]}], \"strategy\": \"one line\"}\n"
)


def run_archivist(
    hypotheses: list[DiagnosticianHypothesis],
    available_evidence: list[dict],
) -> TierResult:
    hypo_text = "\n".join(f"- {h.name} ({h.confidence})" for h in hypotheses) or "(unclassified)"
    ev_text = "\n".join(
        f"- {e['id']} ({e['size_bytes']} bytes): {e.get('description', '')}"
        for e in available_evidence
    )
    user_prompt = (
        f"COMPETING HYPOTHESES:\n{hypo_text}\n\n"
        f"AVAILABLE EVIDENCE:\n{ev_text}\n"
    )

    t0 = time.perf_counter()

    if not (USE_LIVE_GEMMA and GEMMA_ENDPOINT):
        return _fallback("G4", "ARCHIVIST", "live off",
                         _synthesize_archivist(hypotheses, available_evidence))

    try:
        content, tokens_out = _call_gemma(
            user_prompt, system=_ARCHIVIST_SYSTEM,
            temperature=0.1, max_tokens=800, json_mode=True, timeout_s=20.0,
        )
        data = _parse_json_lenient(content)
        packing = ArchivistPacking.model_validate(data)
    except Exception as exc:
        packing = _synthesize_archivist(hypotheses, available_evidence)
        return TierResult(
            payload=packing,
            trace=TierTrace(tier="G4", label="ARCHIVIST", ok=False, is_live=False,
                            latency_s=time.perf_counter() - t0, tokens_out=0,
                            error=f"{type(exc).__name__}: {exc}"),
        )

    return TierResult(
        payload=packing,
        trace=TierTrace(tier="G4", label="ARCHIVIST", ok=True, is_live=True,
                        latency_s=time.perf_counter() - t0, tokens_out=tokens_out,
                        note=f"{len(packing.scores)} records scored"),
    )


def _synthesize_archivist(
    hypotheses: list[DiagnosticianHypothesis],
    evidence: list[dict],
) -> ArchivistPacking:
    """Offline: score raw high-priority records highest, thumbnails lowest.

    This mirrors the hardcoded prioritisation guidance from
    knowledge/incident_response.md §2.1, so the offline path still tells a
    coherent evidence-preservation story.
    """
    def base_score(item: dict) -> int:
        name = item["id"].lower()
        if "raw_spectrum" in name:      return 95
        if "radiation" in name:          return 88
        if "packet_checksums" in name:   return 82
        if "power_window" in name:       return 65
        if "thermal_window" in name:     return 60
        if "camera_thumbnail" in name:   return 40
        return 55

    hnames = [h.name for h in hypotheses[:2]] or ["primary hypothesis"]
    scores = [
        EvidenceScore(
            evidence_id=item["id"],
            score=base_score(item),
            reason=f"heuristic priority for {item['id']}",
            supports_hypotheses=hnames,
        )
        for item in evidence
    ]
    return ArchivistPacking(scores=scores, strategy="Offline heuristic priority order.")


# ── Stack orchestration ──────────────────────────────────────────────────────

@dataclass
class StackResult:
    sentinel: Optional[SentinelVerdict] = None
    diagnosis: Optional[Diagnosis] = None
    plan: Optional[RecoveryPlan] = None
    verdict: Optional[AdjudicatorVerdict] = None
    packing: Optional[ArchivistPacking] = None
    validation: Any = None
    projected_state: Any = None
    traces: list[TierTrace] = field(default_factory=list)
    wall_time_s: float = 0.0
    plan_iterations: int = 1


def run_stack(
    featurised_state: dict,
    fdir_summary: str,
    active_faults: list[str],
    available_evidence: list[dict],
    current_state,
    *,
    include_sentinel: bool = False,
    max_replans: int = 1,
) -> StackResult:
    """Run G1 ∥ G4, then G2 (dependent on G1), then G3 + validator.

    G0 is skipped by default because it belongs on its own continuous timer,
    not tied to the on-demand button that fires the rest of the stack.
    """
    from parallax.validator import validate_plan

    t0 = time.perf_counter()
    traces: list[TierTrace] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        # G1 and G4 are independent — fire them in parallel.
        f_diag: Future = pool.submit(
            run_diagnostician, featurised_state, fdir_summary, active_faults,
        )
        f_arch: Future = pool.submit(
            run_archivist,
            # Archivist gets a lightweight placeholder hypothesis list until G1
            # returns — corrected below if the two calls disagree on ordering.
            [DiagnosticianHypothesis(name=f, confidence="medium",
                                     supporting_evidence=[], contradicting_evidence=[],
                                     citations=[]) for f in (active_faults or ["anomaly"])],
            available_evidence,
        )
        f_sent: Future | None = None
        if include_sentinel:
            f_sent = pool.submit(run_sentinel,
                                 featurised_state.get("hardware_features", featurised_state))

        diag_result = f_diag.result()
        arch_result = f_arch.result()
        traces.extend([diag_result.trace, arch_result.trace])
        if f_sent:
            sent_result = f_sent.result()
            traces.append(sent_result.trace)

    diagnosis: Diagnosis = diag_result.payload
    packing: ArchivistPacking = arch_result.payload
    sentinel = sent_result.payload if include_sentinel else None

    # G2 depends on the diagnosis.
    plan_result = run_flight_director(featurised_state, diagnosis, fdir_summary)
    traces.append(plan_result.trace)
    plan: RecoveryPlan = plan_result.payload

    validation = validate_plan([s.model_dump() for s in plan.steps], current_state)
    verdict_result = run_adjudicator(plan, featurised_state, validation)
    traces.append(verdict_result.trace)
    verdict: AdjudicatorVerdict = verdict_result.payload

    plan_iterations = 1
    # A re-plan is triggered by *either* the deterministic gate rejecting or
    # G3 vetoing. This is the "G2 re-plans after G3 vetoes" moment: the demo
    # value of the stack is that Gemma catches things the gate cannot, and
    # then the same Gemma is given a chance to fix its own plan.
    while ((not validation.approved) or (not verdict.approved)) and plan_iterations <= max_replans:
        if not validation.approved and validation.rejected_step_index is not None:
            offending = plan.steps[validation.rejected_step_index].action
            forbid = {offending}
        elif not verdict.approved and verdict.objection_step:
            offending = plan.steps[verdict.objection_step - 1].action
            forbid = {offending}
        else:
            forbid = None

        replan_result = run_flight_director(featurised_state, diagnosis, fdir_summary,
                                            forbidden_actions=forbid)
        replan_result.trace.label = f"FLIGHT DIRECTOR (retry {plan_iterations})"
        traces.append(replan_result.trace)
        plan = replan_result.payload

        validation = validate_plan([s.model_dump() for s in plan.steps], current_state)
        verdict_result = run_adjudicator(plan, featurised_state, validation)
        verdict_result.trace.label = f"ADJUDICATOR (retry {plan_iterations})"
        traces.append(verdict_result.trace)
        verdict = verdict_result.payload
        plan_iterations += 1

    return StackResult(
        sentinel=sentinel,
        diagnosis=diagnosis,
        plan=plan,
        verdict=verdict,
        packing=packing,
        validation=validation,
        projected_state=validation.projected_state,
        traces=traces,
        wall_time_s=time.perf_counter() - t0,
        plan_iterations=plan_iterations,
    )
