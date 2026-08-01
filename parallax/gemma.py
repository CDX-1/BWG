import json
import os
import re
import urllib.request
import urllib.error
import streamlit as st

from parallax.config import GEMMA_ENDPOINT, API_KEY, USE_LIVE_GEMMA, MODEL_NAME, CACHED_OUTPUTS_DIR
from parallax.models import GemmaPredictiveAnalysis, ParallaxAnalysis, PredictedFailure

PROMPT_TEMPLATE = """You are PARALLAX, an evidence-preservation assistant for a deep-space spacecraft operating under communication delay.

Your role is NOT to conclusively diagnose an unexplained event.

Your role is to:
1. Generate plausible competing explanations.
2. State evidence supporting and contradicting each explanation.
3. Identify which available records must be preserved so investigators can distinguish between the explanations later.
4. Warn when compression, averaging, interpolation, or summarization could destroy relevant evidence.
5. Recommend one safe follow-up observation.
6. Preserve uncertainty explicitly.

Rules:
- Do not invent telemetry.
- Refer only to evidence included in the input or retrieved context.
- Never report an ambiguous event as confirmed.
- Prefer raw observations when derived summaries would remove temporal, spectral, sequencing, or cross-sensor information.
- Consider hardware fault, environmental event, software error, calibration error, and data corruption when supported.
- The total transmission capacity is limited.
- Do not perform byte-budget arithmetic.
- Return output matching the required JSON structure exactly.

SCENARIO:
{scenario_json}

RETRIEVED KNOWLEDGE:
{retrieved_context}

AVAILABLE EVIDENCE:
{evidence_list}

Return ONLY valid JSON matching this exact schema (no markdown, no explanation):
{{
  "event_summary": "...",
  "resolution_status": "unresolved",
  "hypotheses": [
    {{
      "name": "...",
      "confidence": "low|medium|high",
      "supporting_evidence": ["evidence_id", ...],
      "contradicting_evidence": ["evidence_id", ...],
      "evidence_needed": ["evidence_id", ...]
    }}
  ],
  "preservation_priorities": [
    {{
      "evidence_id": "...",
      "priority": 1-5,
      "reason": "...",
      "supports_hypotheses": ["hypothesis name", ...],
      "consequence_if_lost": "..."
    }}
  ],
  "recommended_follow_up": "...",
  "compression_warning": "...",
  "uncertainty_statement": "...",
  "source_references": [
    {{"source_id": "...", "relevance": "..."}}
  ]
}}
"""


PREDICTIVE_PROMPT_TEMPLATE = """You are PARALLAX, an AI mission intelligence system for a deep-space spacecraft.

FDIR (Fault Detection, Isolation, Recovery) has just processed a fault event. Your role is to:
1. Assess the current spacecraft health state after FDIR recovery actions
2. Predict the NEXT most likely failure based on current sensor trends and fault interactions
3. Identify cascading failure risks (what the current fault makes more likely)
4. Recommend 3-5 specific, actionable crew or ground commands
5. Write a concise Earth transmission report for mission control

Rules:
- Be specific about timelines (e.g. "4.2 hours", "72 hours", not "soon")
- Consider physical causality — power faults stress batteries, thermal faults stress electronics, ADCS faults reduce solar input
- Never invent sensor readings not present in the input
- The spacecraft has a 38-minute Earth communication delay — recommendations must be executable autonomously
- Return ONLY valid JSON matching the schema exactly

SPACECRAFT STATE:
{state_json}

FDIR REPORT:
{fdir_json}

MISSION CONTEXT:
{mission_context}

OPERATOR-DEFINED EVENT CONTEXT:
{fault_context}

Return ONLY valid JSON matching this exact schema:
{{
  "current_assessment": "...",
  "system_stability": "stable|degraded|critical|unknown",
  "predicted_failures": [
    {{
      "subsystem": "...",
      "failure_mode": "...",
      "estimated_time_to_failure": "...",
      "probability": "low|medium|high",
      "early_warning_signs": ["...", "..."]
    }}
  ],
  "cascading_risks": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "earth_report": "...",
  "confidence": "low|medium|high"
}}
"""


def _build_prompt(scenario: dict, retrieved_context: list[dict]) -> str:
    evidence_list = "\n".join(
        f"- {e['id']} ({e['size_bytes']} bytes): {e['description']}"
        for e in scenario.get("available_evidence", [])
    )
    context_text = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in retrieved_context
    ) or "No additional context retrieved."

    return PROMPT_TEMPLATE.format(
        scenario_json=json.dumps(scenario, indent=2),
        retrieved_context=context_text,
        evidence_list=evidence_list,
    )


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
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"]


def _parse_response(text: str) -> ParallaxAnalysis:
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    data = json.loads(text)
    return ParallaxAnalysis.model_validate(data)


def stream_gemma(prompt: str):
    """Generator that yields string chunks from a streaming Gemma response."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": True,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(GEMMA_ENDPOINT, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, IndexError, KeyError):
                continue


def build_prompt(scenario: dict, retrieved_context: list[dict]) -> str:
    """Public wrapper around _build_prompt."""
    return _build_prompt(scenario, retrieved_context)


def parse_response(text: str) -> ParallaxAnalysis:
    """Public wrapper around _parse_response."""
    return _parse_response(text)


def load_cached_result(scenario_id: str) -> ParallaxAnalysis:
    path = os.path.join(CACHED_OUTPUTS_DIR, f"{scenario_id}.json")
    with open(path) as f:
        data = json.load(f)
    return ParallaxAnalysis.model_validate(data)


def run_parallax_analysis(
    scenario: dict,
    retrieved_context: list[dict],
) -> tuple[ParallaxAnalysis, bool]:
    """Returns (analysis, is_live) where is_live=False means fallback was used."""
    scenario_id = scenario.get("scenario_id", "unknown")

    if USE_LIVE_GEMMA and GEMMA_ENDPOINT:
        prompt = _build_prompt(scenario, retrieved_context)
        try:
            raw = _call_gemma_api(prompt)
            analysis = _parse_response(raw)
            return analysis, True
        except Exception as e:
            st.warning(f"Live Gemma call failed ({e}). Retrying once...")
            try:
                raw = _call_gemma_api(prompt)
                analysis = _parse_response(raw)
                return analysis, True
            except Exception:
                pass

    # Fallback to cached output
    try:
        analysis = load_cached_result(scenario_id)
        return analysis, False
    except Exception as e:
        raise RuntimeError(f"No cached output available for {scenario_id}: {e}")


def run_predictive_analysis(
    state_dict: dict,
    fdir_dict: dict,
    mission_context: str,
    fault_id: str,
    fault_context: dict | None = None,
    stream_callback=None,
) -> tuple["GemmaPredictiveAnalysis", bool]:
    prompt = PREDICTIVE_PROMPT_TEMPLATE.format(
        state_json=json.dumps(state_dict, indent=2),
        fdir_json=json.dumps(fdir_dict, indent=2),
        mission_context=mission_context,
        fault_context=json.dumps(fault_context or {"source": "preconfigured FDIR fault"}, indent=2),
    )

    if USE_LIVE_GEMMA and GEMMA_ENDPOINT:
        try:
            if stream_callback is not None:
                collected = []
                for chunk in stream_gemma(prompt):
                    collected.append(chunk)
                    stream_callback("".join(collected))
                raw = "".join(collected)
            else:
                raw = _call_gemma_api(prompt)
            text = re.sub(r"^```(?:json)?\n?", "", raw.strip())
            text = re.sub(r"\n?```$", "", text.strip())
            analysis = GemmaPredictiveAnalysis.model_validate(json.loads(text))
            return analysis, True
        except Exception:
            pass

    # Cached predictions are only valid for the isolated, named demo cases.
    # Runtime event context must never be replaced with a scripted result.
    if fault_context:
        return _dynamic_predictive_fallback(state_dict, fdir_dict, fault_context), False

    try:
        path = os.path.join(CACHED_OUTPUTS_DIR, f"predict_{fault_id}.json")
        with open(path) as f:
            data = json.load(f)
        return GemmaPredictiveAnalysis.model_validate(data), False
    except Exception as e:
        raise RuntimeError(f"No cached prediction for {fault_id}: {e}")


def _dynamic_predictive_fallback(
    state_dict: dict,
    fdir_dict: dict,
    fault_context: dict,
) -> GemmaPredictiveAnalysis:
    """Honest offline behaviour for a runtime event without a canned response.

    This is intentionally a conservative state summary, not a replacement for
    Gemma. It keeps custom event demos usable when hosted inference is offline
    and clearly leaves diagnosis unresolved.
    """
    title = fault_context.get("label", "Operator-defined anomaly")
    description = fault_context.get("description", "No description supplied.")
    affected = fault_context.get("subsystems", [])
    severity = fault_context.get("severity", "warning")
    health = state_dict.get("subsystem_health", {})
    degraded = [name for name, status in health.items() if status != "nominal"]
    stability = "critical" if severity == "critical" or "failed" in health.values() else "degraded"
    predicted = [
        PredictedFailure(
            subsystem=subsystem,
            failure_mode=f"Potential progression of {title.lower()}",
            estimated_time_to_failure="Unknown — trend data required",
            probability="high" if severity == "critical" else "medium",
            early_warning_signs=["Change in the supplied telemetry override", "Additional FDIR threshold crossings"],
        )
        for subsystem in affected[:3]
    ]
    return GemmaPredictiveAnalysis(
        current_assessment=(
            f"Offline dynamic assessment for {title}: {description} "
            f"Affected subsystems: {', '.join(affected)}. The condition remains unclassified; preserve raw telemetry."
        ),
        system_stability=stability,
        predicted_failures=predicted,
        cascading_risks=[
            f"Monitor {subsystem} for coupling with {', '.join(degraded) or 'other spacecraft systems'}."
            for subsystem in affected[:2]
        ],
        recommended_actions=[
            "Preserve the high-rate telemetry window and command history before any reset.",
            "Hold irreversible recovery commands until the event has been reviewed against operating procedures.",
            "Request a hosted Gemma assessment when the communications path is available.",
        ],
        earth_report=(
            f"PRIORITY REPORT — OPERATOR-DEFINED EVENT\n\nEVENT: {title}\nOBSERVATION: {description}\n"
            f"AFFECTED SYSTEMS: {', '.join(affected)}\nSTATUS: {fdir_dict.get('summary', 'FDIR review pending')}\n\n"
            "This event has not been conclusively diagnosed. Raw telemetry and command history have been retained for ground analysis."
        ),
        confidence="low",
    )
