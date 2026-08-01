import json
import os
import re
import urllib.request
import urllib.error
import streamlit as st

from parallax.config import GEMMA_ENDPOINT, API_KEY, USE_LIVE_GEMMA, MODEL_NAME, CACHED_OUTPUTS_DIR
from parallax.models import ParallaxAnalysis

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
            # Cache the successful output
            os.makedirs(CACHED_OUTPUTS_DIR, exist_ok=True)
            cache_path = os.path.join(CACHED_OUTPUTS_DIR, f"{scenario_id}.json")
            with open(cache_path, "w") as f:
                f.write(analysis.model_dump_json(indent=2))
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
