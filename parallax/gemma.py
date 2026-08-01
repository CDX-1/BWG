import json
import os
import re
import urllib.request
import urllib.error
import streamlit as st

from parallax.config import GEMMA_ENDPOINT, API_KEY, USE_LIVE_GEMMA, MODEL_NAME, CACHED_OUTPUTS_DIR
from parallax.models import GemmaPredictiveAnalysis, ParallaxAnalysis, PredictedFailure, FixOption
from parallax.spacecraft import FAULT_DEFINITIONS

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

FDIR (Fault Detection, Isolation, Recovery) has just executed a basic autonomous recovery. Your role is to:
1. Assess the current spacecraft health state after FDIR recovery actions
2. Estimate the Earth communication delay impact on the decision window
3. Calculate the probability that the basic FDIR fix succeeds long-term (0-100 integer)
4. Calculate the probability of a cascade failure occurring (0-100 integer)
5. Propose 3 alternative fix strategies with individual success probabilities
6. Choose the best fix option and decide whether to execute it immediately (only if >75% success and autonomous)
7. Predict the NEXT most likely failure based on sensor trends
8. Write a concise Earth transmission report

Rules:
- Be specific about timelines (e.g. "4.2 hours", "72 hours", not "soon")
- Consider physical causality — power faults stress batteries, thermal faults stress electronics, ADCS faults reduce solar input
- Never invent sensor readings not present in the input
- The spacecraft has a 38-minute Earth communication delay — only recommend autonomous fixes
- Several faults may be active AT THE SAME TIME. When they are, assess the combined state, not each fault in isolation: identify how the faults interact and compete for the same power, thermal and pointing margin, note where one fault removes the backup path another recovery depends on, and lower the success probabilities accordingly
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
  "earth_delay_min": 38,
  "basic_fix_success_pct": <integer 0-100>,
  "cascade_failure_pct": <integer 0-100>,
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
  "alternative_fixes": [
    {{
      "name": "...",
      "description": "...",
      "success_pct": <integer 0-100>,
      "risk": "low|medium|high",
      "autonomous": true|false
    }}
  ],
  "chosen_fix": "...",
  "execute_immediately": true|false,
  "execute_reason": "...",
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
    fault_id: str | list[str],
    fault_context: dict | None = None,
    stream_callback=None,
) -> tuple["GemmaPredictiveAnalysis", bool]:
    """Analyse the spacecraft under one or several simultaneously active faults.

    `fault_id` accepts a single id or a list of concurrently active ids.
    """
    fault_ids = [fault_id] if isinstance(fault_id, str) else list(fault_id)
    fault_ids = [f for f in fault_ids if f]

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
        primary = fault_ids[0] if fault_ids else ""
        return _dynamic_predictive_fallback(state_dict, fdir_dict, fault_context, primary), False

    analyses = []
    missing = []
    for fid in fault_ids:
        try:
            analyses.append((fid, _load_cached_prediction(fid)))
        except Exception:
            missing.append(fid)

    if not analyses:
        raise RuntimeError(f"No cached prediction for {', '.join(fault_ids) or 'unknown fault'}")
    if len(analyses) == 1 and not missing:
        return analyses[0][1], False

    return _combine_predictions(analyses, state_dict, fdir_dict), False


def _load_cached_prediction(fault_id: str) -> GemmaPredictiveAnalysis:
    path = os.path.join(CACHED_OUTPUTS_DIR, f"predict_{fault_id}.json")
    with open(path) as f:
        data = json.load(f)
    analysis = GemmaPredictiveAnalysis.model_validate(data)
    if not analysis.alternative_fixes:
        analysis = _augment_with_fixes(analysis, fault_id)
    return analysis


_FALLBACK_FIXES: dict[str, list[FixOption]] = {
    "solar_string_loss": [
        FixOption(name="Load Shedding (Basic FDIR)", description="Activate battery conservation, reduce non-essential power loads 40%", success_pct=72, risk="low", autonomous=True),
        FixOption(name="Array Reorientation", description="Adjust solar array angle +2.3° toward sun vector for maximum exposure", success_pct=81, risk="medium", autonomous=True),
        FixOption(name="String Bypass Relay", description="Activate bypass relay to route power around failed string 2 via strings 1 and 3", success_pct=88, risk="high", autonomous=True),
    ],
    "pcu_fault": [
        FixOption(name="Backup PCU Switch (Basic FDIR)", description="Switch to redundant PCU-A module, isolate and power-down failed PCU-B", success_pct=78, risk="low", autonomous=True),
        FixOption(name="Load Reduction Cycle", description="Step-reduce power draw 30%, allow PCU thermal recovery over 2 hours", success_pct=64, risk="medium", autonomous=True),
        FixOption(name="Full Bus Reset", description="Execute full power bus reset sequence with PCU cold restart — requires 800ms blackout", success_pct=91, risk="high", autonomous=False),
    ],
    "reaction_wheel_fault": [
        FixOption(name="Torquer Backup (Basic FDIR)", description="Engage magnetic torquers as ADCS backup, command RW-2 spin-down", success_pct=76, risk="low", autonomous=True),
        FixOption(name="RW Desaturation", description="Cross-desaturate remaining three wheels, redistribute stored angular momentum", success_pct=83, risk="medium", autonomous=True),
        FixOption(name="Sun-Safe Slew", description="Enter Sun-pointing safe mode, suspend science pointing until RW-2 is assessed", success_pct=95, risk="low", autonomous=True),
    ],
    "spectrometer_fault": [
        FixOption(name="Power Cycle (Basic FDIR)", description="Power cycle spectrometer detector array and flush instrument buffers", success_pct=67, risk="low", autonomous=True),
        FixOption(name="Detector Reset Sequence", description="Execute full detector reset with calibration lamp check and dark-frame test", success_pct=74, risk="medium", autonomous=True),
        FixOption(name="Instrument Safe Mode", description="Place spectrometer in safe mode, preserve all raw science data, await ground review", success_pct=99, risk="low", autonomous=True),
    ],
    "comms_dropout": [
        FixOption(name="LGA Fallback (Basic FDIR)", description="Switch to Low-Gain Antenna LGA-2 for contingency comms at 115 kbps", success_pct=89, risk="low", autonomous=True),
        FixOption(name="HGA Autonomous Repoint", description="Execute HGA repoint sequence using onboard star tracker attitude data", success_pct=71, risk="medium", autonomous=True),
        FixOption(name="Earth Acquisition Scan", description="Run progressive antenna scan ±15° around predicted Earth direction", success_pct=82, risk="medium", autonomous=True),
    ],
    "thermal_runaway": [
        FixOption(name="Emergency Cooling (Basic FDIR)", description="Deploy radiator panels, activate emergency thermal control loops", success_pct=58, risk="low", autonomous=True),
        FixOption(name="Load Reduction", description="Reduce PCU computational load 30%, disable non-essential heaters", success_pct=68, risk="low", autonomous=True),
        FixOption(name="Thermal Safe Mode", description="All payload off, passive cooling only — halts science operations", success_pct=94, risk="medium", autonomous=True),
    ],
}

_FALLBACK_METRICS: dict[str, dict] = {
    "solar_string_loss":    {"basic_fix_success_pct": 72, "cascade_failure_pct": 28},
    "pcu_fault":            {"basic_fix_success_pct": 65, "cascade_failure_pct": 45},
    "reaction_wheel_fault": {"basic_fix_success_pct": 76, "cascade_failure_pct": 31},
    "spectrometer_fault":   {"basic_fix_success_pct": 67, "cascade_failure_pct": 18},
    "comms_dropout":        {"basic_fix_success_pct": 89, "cascade_failure_pct": 22},
    "thermal_runaway":      {"basic_fix_success_pct": 58, "cascade_failure_pct": 62},
}


def _augment_with_fixes(analysis: GemmaPredictiveAnalysis, fault_id: str) -> GemmaPredictiveAnalysis:
    fixes = _FALLBACK_FIXES.get(fault_id, [])
    metrics = _FALLBACK_METRICS.get(fault_id, {})
    autonomous_fixes = [f for f in fixes if f.autonomous]
    best = max(autonomous_fixes, key=lambda f: f.success_pct, default=None)
    chosen = best.name if best else "Basic FDIR Recovery"
    execute = bool(best and best.success_pct >= 75)
    reason = (f"Highest autonomous success probability: {best.success_pct}% with {best.risk} risk." if best else "")
    return analysis.model_copy(update={
        "alternative_fixes": fixes,
        "chosen_fix": chosen,
        "execute_immediately": execute,
        "execute_reason": reason,
        "basic_fix_success_pct": metrics.get("basic_fix_success_pct", analysis.basic_fix_success_pct),
        "cascade_failure_pct": metrics.get("cascade_failure_pct", analysis.cascade_failure_pct),
        "earth_delay_min": 38,
    })


_STABILITY_RANK = {"stable": 0, "unknown": 1, "degraded": 2, "critical": 3}
# Shared low/medium/high scale, used for both confidence and probability.
_LEVEL_RANK = {"high": 2, "medium": 1, "low": 0}


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _fault_label(fault_id: str) -> str:
    definition = FAULT_DEFINITIONS.get(fault_id, {})
    return definition.get("label", fault_id.replace("_", " ").title())


def _combine_predictions(
    analyses: list[tuple[str, GemmaPredictiveAnalysis]],
    state_dict: dict,
    fdir_dict: dict,
) -> GemmaPredictiveAnalysis:
    """Merge the per-fault cached analyses into one compound-fault assessment.

    The offline path has no cached output for arbitrary fault combinations, so
    the individual analyses are composed probabilistically: each recovery must
    independently succeed, and any one cascade is enough to lose the mission.
    """
    labels = [_fault_label(fid) for fid, _ in analyses]
    parts = [analysis for _, analysis in analyses]

    basic_success = 1.0
    no_cascade = 1.0
    for analysis in parts:
        basic_success *= analysis.basic_fix_success_pct / 100
        no_cascade *= 1 - (analysis.cascade_failure_pct / 100)
    basic_fix_success_pct = max(1, round(basic_success * 100))
    cascade_failure_pct = min(97, round((1 - no_cascade) * 100))

    health = state_dict.get("subsystem_health", {})
    failed = [name for name, status in health.items() if status == "failed"]
    stability = max((a.system_stability for a in parts), key=lambda s: _STABILITY_RANK.get(s, 1))
    if len(failed) > 1:
        stability = "critical"

    # One best autonomous option per fault, plus a whole-spacecraft fallback.
    fixes: list[FixOption] = []
    for fid, analysis in analyses:
        candidates = [f for f in analysis.alternative_fixes if f.autonomous] or analysis.alternative_fixes
        if not candidates:
            continue
        best = max(candidates, key=lambda f: f.success_pct)
        fixes.append(best.model_copy(update={
            "name": f"{_fault_label(fid)}: {best.name}",
            "success_pct": max(1, round(best.success_pct * (0.85 ** (len(parts) - 1)))),
        }))
    fixes.append(FixOption(
        name="Staged Safe Mode Entry",
        description=(
            f"Suspend science, hold Sun-pointing and recover {len(parts)} faults sequentially "
            f"in power-priority order rather than concurrently."
        ),
        success_pct=max(basic_fix_success_pct, 84),
        risk="medium",
        autonomous=True,
    ))

    best_overall = max([f for f in fixes if f.autonomous] or fixes, key=lambda f: f.success_pct)

    # Merged lists are capped so the compound view stays readable in the UI.
    predicted: list[PredictedFailure] = []
    seen = set()
    for analysis in parts:
        for failure in analysis.predicted_failures:
            key = (failure.subsystem, failure.failure_mode)
            if key not in seen:
                seen.add(key)
                predicted.append(failure)
    predicted.sort(key=lambda f: -_LEVEL_RANK.get(f.probability, 0))
    predicted = predicted[:6]

    risks = _dedupe([risk for analysis in parts for risk in analysis.cascading_risks])[:6]
    risks.insert(0, (
        f"{len(parts)} concurrent faults ({', '.join(labels)}) share one power and thermal budget — "
        f"recovery sequences compete for margin that is sized for a single failure."
    ))
    if len(failed) > 1:
        risks.insert(1, (
            f"Failed subsystems {', '.join(failed)} remove each other's backup path; "
            f"no cross-subsystem redundancy remains."
        ))

    actions = [
        f"Sequence recovery by power priority — do not run {len(parts)} recovery paths concurrently.",
        *_dedupe([action for analysis in parts for action in analysis.recommended_actions])[:6],
    ]

    assessment = (
        f"{len(parts)} faults are active simultaneously: {', '.join(labels)}. "
        f"Combined basic-FDIR success falls to {basic_fix_success_pct}% because each recovery must hold "
        f"independently, and cascade risk rises to {cascade_failure_pct}%. "
        + " ".join(a.current_assessment for a in parts)
    )

    report_sections = "\n\n".join(
        f"— {label} —\n{analysis.earth_report}" for label, (_, analysis) in zip(labels, analyses)
    )
    earth_report = (
        f"PRIORITY REPORT — COMPOUND ANOMALY ({len(parts)} CONCURRENT FAULTS)\n\n"
        f"FAULTS: {', '.join(labels)}\n"
        f"FAILED SUBSYSTEMS: {', '.join(failed) or 'none'}\n"
        f"STATUS: {fdir_dict.get('summary', 'FDIR review pending')}\n"
        f"COMBINED FDIR SUCCESS: {basic_fix_success_pct}%  ·  CASCADE RISK: {cascade_failure_pct}%\n\n"
        f"{report_sections}"
    )

    # Composing single-fault analyses is weaker evidence than a purpose-built
    # assessment of the combination, so never claim high confidence here.
    confidence = min((a.confidence for a in parts), key=lambda c: _LEVEL_RANK.get(c, 0))
    if confidence == "high":
        confidence = "medium"

    return GemmaPredictiveAnalysis(
        current_assessment=assessment,
        system_stability=stability,
        earth_delay_min=parts[0].earth_delay_min,
        basic_fix_success_pct=basic_fix_success_pct,
        cascade_failure_pct=cascade_failure_pct,
        predicted_failures=predicted,
        cascading_risks=risks,
        alternative_fixes=fixes,
        chosen_fix=best_overall.name,
        execute_immediately=best_overall.autonomous and best_overall.success_pct >= 75,
        execute_reason=(
            f"Best autonomous option across {len(parts)} concurrent faults: "
            f"{best_overall.success_pct}% with {best_overall.risk} risk."
        ),
        recommended_actions=actions,
        earth_report=earth_report,
        confidence=confidence,
    )


def _dynamic_predictive_fallback(
    state_dict: dict,
    fdir_dict: dict,
    fault_context: dict,
    fault_id: str = "",
) -> GemmaPredictiveAnalysis:
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
    fixes = _FALLBACK_FIXES.get(fault_id, [
        FixOption(name="Basic FDIR Recovery", description="Execute deterministic recovery sequence for affected subsystems", success_pct=65, risk="low", autonomous=True),
        FixOption(name="Safe Mode Entry", description="Enter spacecraft safe mode: minimal power, Sun-pointing, preserve all data", success_pct=90, risk="low", autonomous=True),
        FixOption(name="Full Subsystem Reset", description="Power cycle all affected subsystems and re-initialise from last known-good state", success_pct=72, risk="high", autonomous=True),
    ])
    metrics = _FALLBACK_METRICS.get(fault_id, {"basic_fix_success_pct": 65, "cascade_failure_pct": 35})
    autonomous_fixes = [f for f in fixes if f.autonomous]
    best = max(autonomous_fixes, key=lambda f: f.success_pct, default=None)
    chosen = best.name if best else "Basic FDIR Recovery"
    execute = bool(best and best.success_pct >= 75)
    reason = (f"Highest autonomous success probability: {best.success_pct}% with {best.risk} risk." if best else "")
    return GemmaPredictiveAnalysis(
        current_assessment=(
            f"Offline assessment for {title}: {description} "
            f"Affected: {', '.join(affected)}. Condition unclassified — preserve raw telemetry."
        ),
        system_stability=stability,
        earth_delay_min=38,
        basic_fix_success_pct=metrics["basic_fix_success_pct"],
        cascade_failure_pct=metrics["cascade_failure_pct"],
        predicted_failures=predicted,
        cascading_risks=[
            f"Monitor {subsystem} for coupling with {', '.join(degraded) or 'other systems'}."
            for subsystem in affected[:2]
        ],
        alternative_fixes=fixes,
        chosen_fix=chosen,
        execute_immediately=execute,
        execute_reason=reason,
        recommended_actions=[
            "Preserve high-rate telemetry window and command history before any reset.",
            "Hold irreversible commands until event reviewed against operating procedures.",
            "Request Gemma assessment when communications path is available.",
        ],
        earth_report=(
            f"PRIORITY REPORT — OPERATOR EVENT\n\nEVENT: {title}\nOBSERVATION: {description}\n"
            f"AFFECTED: {', '.join(affected)}\nSTATUS: {fdir_dict.get('summary', 'FDIR review pending')}\n\n"
            "Event not conclusively diagnosed. Raw telemetry retained for ground analysis."
        ),
        confidence="low",
    )
