"""Streamlit renderers for the five-tier Gemma stack.

Kept in its own module so app.py stays focused on layout and the tier-specific
presentation logic can evolve without churning the rest of the dashboard.
"""

from __future__ import annotations

import streamlit as st

from parallax.baseline import naive_capsule
from parallax.capsule import build_capsule
from parallax.models import (
    AdjudicatorVerdict,
    ArchivistPacking,
    Diagnosis,
    EvidenceAssessment,
    RecoveryPlan,
    SentinelVerdict,
    TierTrace,
)
from parallax.tiers import StackResult
from parallax.validator import ValidationReport


# ── Palette echoes app.py ───────────────────────────────────────────────────

WHITE   = "#ffffff"
BORDER  = "#e2e8f0"
BORDER_S= "#cbd5e1"
TEXT    = "#1e293b"
TEXT_M  = "#64748b"
TEXT_D  = "#94a3b8"
NAVY    = "#1e3a6e"
BLUE    = "#2563eb"
GREEN   = "#059669"
RED     = "#dc2626"
AMBER   = "#d97706"
PURPLE  = "#7c3aed"

TIER_COLORS = {
    "G0": PURPLE,
    "G1": BLUE,
    "G2": NAVY,
    "G3": AMBER,
    "G4": GREEN,
}

TIER_LABELS = {
    "G0": "SENTINEL",
    "G1": "DIAGNOSTICIAN",
    "G2": "FLIGHT DIRECTOR",
    "G3": "ADJUDICATOR",
    "G4": "ARCHIVIST",
}

HEALTH_COLOR = {"nominal": GREEN, "degraded": AMBER, "failed": RED, "recovering": BLUE}
HEALTH_ICON  = {"nominal": "●", "degraded": "◑", "failed": "○", "recovering": "↻"}
HEALTH_SCORE = {"nominal": 3, "recovering": 2, "degraded": 1, "failed": 0}


# ── Small building blocks ───────────────────────────────────────────────────

def _card(body, left_color=None, bg=WHITE, padding="12px 14px"):
    ls = f"border-left: 4px solid {left_color};" if left_color else ""
    return (f'<div style="background:{bg};border:1px solid {BORDER};border-radius:10px;'
            f'padding:{padding};margin:4px 0;box-shadow:0 2px 6px rgba(0,0,0,0.06);{ls}">'
            f'{body}</div>')


def _badge(text, color, sm=False):
    sz = "0.7em" if sm else "0.76em"
    return (f'<span style="background:{color}15;border:1px solid {color}40;color:{color};'
            f'padding:2px 9px;border-radius:20px;font-size:{sz};font-weight:700;">{text}</span>')


def _pill(text, color):
    return (f'<span style="background:{color}18;border:1px solid {color}55;color:{color};'
            f'padding:1px 8px;border-radius:6px;font-size:0.68em;font-weight:700;'
            f'font-family:ui-monospace,Menlo,monospace;">{text}</span>')


# ── Pipeline strip (top of the panel) ───────────────────────────────────────

def render_pipeline_strip(traces: list[TierTrace], wall_time_s: float, plan_iterations: int):
    """Compact five-icon strip showing what actually ran in this stack pass."""
    if not traces:
        return
    tier_order = ["G0", "G1", "G2", "G3", "G4"]
    latest_by_tier: dict[str, TierTrace] = {}
    for tr in traces:
        latest_by_tier[tr.tier] = tr

    cells = []
    for tier in tier_order:
        tr = latest_by_tier.get(tier)
        color = TIER_COLORS[tier]
        if tr is None:
            cells.append(
                f'<div style="flex:1;background:{WHITE};border:1px dashed {BORDER_S};'
                f'border-radius:8px;padding:8px;text-align:center;">'
                f'<div style="color:{TEXT_D};font-size:0.65em;font-weight:800;'
                f'letter-spacing:0.09em;">{tier}</div>'
                f'<div style="color:{TEXT_D};font-size:0.68em;margin-top:2px;">— idle —</div>'
                f'</div>'
            )
            continue
        edge = color if tr.ok else RED
        badge_bg = ("LIVE" if tr.is_live else "CACHED")
        badge_color = GREEN if tr.is_live else AMBER
        if not tr.ok:
            badge_bg, badge_color = "ERROR", RED
        latency = f"{tr.latency_s:.2f}s" if tr.latency_s > 0 else "—"
        cells.append(
            f'<div style="flex:1;background:{WHITE};border:1px solid {edge}55;border-top:3px solid {edge};'
            f'border-radius:8px;padding:8px 10px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
            f'<div style="color:{color};font-size:0.7em;font-weight:800;letter-spacing:0.09em;">'
            f'{tier} · {TIER_LABELS[tier]}</div>'
            f'<div style="margin-top:4px;">'
            f'<span style="font-size:0.65em;color:{badge_color};font-weight:700;">{badge_bg}</span>'
            f'&nbsp;&middot;&nbsp;<span style="color:{TEXT_M};font-size:0.68em;font-family:monospace;">{latency}</span>'
            f'&nbsp;&middot;&nbsp;<span style="color:{TEXT_D};font-size:0.65em;font-family:monospace;">{tr.tokens_out}t</span>'
            f'</div>'
            f'<div style="color:{TEXT_M};font-size:0.66em;margin-top:3px;line-height:1.3;">{tr.note or ""}</div>'
            + (f'<div style="color:{RED};font-size:0.62em;margin-top:2px;line-height:1.3;">{tr.error}</div>' if tr.error else "")
            + '</div>'
        )

    st.markdown(
        f'<div style="display:flex;gap:6px;">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )

    footnote_bits = [f"stack wall time <strong>{wall_time_s:.2f}s</strong>"]
    if plan_iterations > 1:
        footnote_bits.append(f"<strong>{plan_iterations}</strong> plan iterations (G3 vetoed the first)")
    st.markdown(
        f'<div style="text-align:right;color:{TEXT_D};font-size:0.68em;margin-top:4px;">'
        f'{" &nbsp;·&nbsp; ".join(footnote_bits)}</div>',
        unsafe_allow_html=True,
    )


# ── G1 diagnosis ────────────────────────────────────────────────────────────

def render_diagnosis(diag: Diagnosis):
    if not diag:
        return
    st.markdown(
        f'<div style="color:{TEXT_D};font-size:0.7em;font-weight:800;letter-spacing:0.1em;'
        f'margin:10px 0 4px;">G1 · COMPETING HYPOTHESES</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.75em;color:{TEXT_M};margin-bottom:6px;line-height:1.5;">{diag.summary}</div>',
        unsafe_allow_html=True,
    )
    for h in diag.hypotheses:
        conf_c = {"high": GREEN, "medium": AMBER, "low": TEXT_D}.get(h.confidence, TEXT_D)
        cites = " ".join(_pill(c, NAVY) for c in h.citations) or _pill("uncited", RED)
        supp = ", ".join(h.supporting_evidence) or "—"
        contra = ", ".join(h.contradicting_evidence) or "—"
        st.markdown(_card(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
            f'<span style="font-weight:700;font-size:0.8em;color:{TEXT};">{h.name}</span>'
            f'{_badge(h.confidence.upper(), conf_c, sm=True)}</div>'
            f'<div style="font-size:0.71em;color:{TEXT_M};margin:3px 0;">'
            f'<strong>supports:</strong> {supp}</div>'
            f'<div style="font-size:0.71em;color:{TEXT_M};margin:3px 0;">'
            f'<strong>contradicts:</strong> {contra}</div>'
            f'<div style="margin-top:6px;">{cites}</div>',
            left_color=conf_c,
        ), unsafe_allow_html=True)


# ── G2 plan + validator gates ───────────────────────────────────────────────

def render_plan_and_gates(plan: RecoveryPlan, validation: ValidationReport, verdict: AdjudicatorVerdict):
    if not plan:
        return

    st.markdown(
        f'<div style="color:{TEXT_D};font-size:0.7em;font-weight:800;letter-spacing:0.1em;'
        f'margin:16px 0 4px;">G2 · RECOVERY PLAN</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.75em;color:{TEXT_M};margin-bottom:6px;line-height:1.5;">{plan.summary}</div>',
        unsafe_allow_html=True,
    )

    # Per-step cards, with the gate outcomes for that step laid out inline.
    gate_by_step: dict[int, list] = {}
    for gr in validation.per_step:
        gate_by_step.setdefault(gr.step_index, []).append(gr)

    for idx, step in enumerate(plan.steps):
        step_gates = gate_by_step.get(idx, [])
        rejected_here = validation.rejected_step_index == idx and not validation.approved

        gate_row = []
        for gr in step_gates:
            color = GREEN if gr.passed else RED
            symbol = "✓" if gr.passed else "✗"
            gate_row.append(_pill(f"{symbol} {gr.gate}", color))
        gate_html = " ".join(gate_row)

        params_str = ", ".join(f"{k}={v}" for k, v in (step.params or {}).items())
        header = f"{idx + 1}. {step.action}({params_str})"
        edge = RED if rejected_here else (NAVY if verdict and verdict.approved else AMBER)
        cites = " ".join(_pill(c, NAVY) for c in step.citations) or ""

        st.markdown(_card(
            f'<div style="font-weight:700;font-size:0.8em;color:{TEXT};margin-bottom:4px;'
            f'font-family:ui-monospace,Menlo,monospace;">{header}</div>'
            f'<div style="font-size:0.72em;color:{TEXT_M};margin:4px 0;line-height:1.5;">{step.rationale}</div>'
            f'<div style="margin:6px 0;">{gate_html}</div>'
            + (f'<div style="margin:4px 0;">{cites}</div>' if cites else "")
            + (f'<div style="font-size:0.72em;color:{RED};margin-top:6px;font-weight:600;">'
               f'gate reason: {validation.rejected_reason}</div>' if rejected_here else ""),
            left_color=edge,
        ), unsafe_allow_html=True)

    # Overall validator summary
    val_color = GREEN if validation.approved else RED
    st.markdown(
        f'<div style="background:{val_color}0d;border:1.5px solid {val_color}40;'
        f'border-radius:8px;padding:8px 12px;margin-top:8px;">'
        f'<div style="font-size:0.68em;font-weight:800;color:{TEXT_D};letter-spacing:0.08em;">'
        f'DETERMINISTIC GATES</div>'
        f'<div style="font-size:0.78em;color:{val_color};font-weight:700;">{validation.summary_line}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_adjudicator(verdict: AdjudicatorVerdict):
    if not verdict:
        return
    color = GREEN if verdict.approved else RED
    label = "APPROVED" if verdict.approved else f"VETO · step {verdict.objection_step or '?'}"
    st.markdown(
        f'<div style="background:{color}0d;border:2px solid {color}55;'
        f'border-radius:10px;padding:12px 14px;margin-top:10px;">'
        f'<div style="font-size:0.68em;font-weight:800;color:{TEXT_D};letter-spacing:0.08em;'
        f'margin-bottom:4px;">G3 · ADJUDICATOR VERDICT</div>'
        f'<div style="font-weight:800;color:{color};font-size:0.9em;">{label}</div>'
        f'<div style="font-size:0.74em;color:{TEXT_M};margin-top:5px;line-height:1.5;">{verdict.reason}</div>'
        + ("".join(
            f'<div style="font-size:0.71em;color:{TEXT_M};margin-top:4px;">'
            f'• {rn}</div>' for rn in (verdict.revision_notes or [])
        ))
        + '</div>',
        unsafe_allow_html=True,
    )


# ── Before / after delta ────────────────────────────────────────────────────

def render_before_after(before, after):
    """Subsystem health strip: before-plan vs projected-after-plan."""
    st.markdown(
        f'<div style="color:{TEXT_D};font-size:0.7em;font-weight:800;letter-spacing:0.1em;'
        f'margin:12px 0 6px;">BEFORE / AFTER — SUBSYSTEM HEALTH</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for subsystem, before_h in before.subsystem_health.items():
        after_h = after.subsystem_health.get(subsystem, before_h)
        bc = HEALTH_COLOR.get(before_h, TEXT_D)
        ac = HEALTH_COLOR.get(after_h, TEXT_D)
        arrow_c = GREEN if HEALTH_SCORE.get(after_h, 0) > HEALTH_SCORE.get(before_h, 0) else \
                  (RED if HEALTH_SCORE.get(after_h, 0) < HEALTH_SCORE.get(before_h, 0) else TEXT_D)
        rows.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:4px 8px;border-bottom:1px solid {BORDER};font-size:0.75em;">'
            f'<span style="color:{TEXT};font-weight:600;flex:1.5;">{subsystem}</span>'
            f'<span style="color:{bc};font-family:monospace;flex:1;">{HEALTH_ICON.get(before_h, "?")} {before_h}</span>'
            f'<span style="color:{arrow_c};font-family:monospace;flex:0.4;text-align:center;">→</span>'
            f'<span style="color:{ac};font-family:monospace;flex:1;">{HEALTH_ICON.get(after_h, "?")} {after_h}</span>'
            f'</div>'
        )
    delta_before = sum(HEALTH_SCORE.get(h, 0) for h in before.subsystem_health.values())
    delta_after  = sum(HEALTH_SCORE.get(h, 0) for h in after.subsystem_health.values())
    delta_score  = delta_after - delta_before
    delta_c = GREEN if delta_score > 0 else (RED if delta_score < 0 else TEXT_D)
    st.markdown(_card(
        "".join(rows)
        + f'<div style="margin-top:8px;text-align:right;font-size:0.75em;color:{delta_c};'
        f'font-weight:700;">Health delta: {"+" if delta_score >= 0 else ""}{delta_score}</div>',
        left_color=delta_c or BORDER_S,
    ), unsafe_allow_html=True)


# ── G0 sentinel strip ───────────────────────────────────────────────────────

def render_sentinel_strip(verdict: SentinelVerdict, is_live: bool, latency_s: float):
    if not verdict:
        return
    color = {"nominal": GREEN, "watch": AMBER, "anomalous": RED}.get(verdict.status, TEXT_D)
    live_badge = _badge("LIVE" if is_live else "OFFLINE", GREEN if is_live else TEXT_D, sm=True)
    st.markdown(
        f'<div style="background:{color}0c;border:1px solid {color}55;border-left:5px solid {color};'
        f'border-radius:8px;padding:8px 14px;display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:8px;">'
        f'<div>'
        f'<span style="font-size:0.65em;font-weight:800;color:{TEXT_D};letter-spacing:0.1em;">'
        f'G0 · SENTINEL</span>'
        f'&nbsp;&nbsp;<span style="font-size:0.85em;font-weight:800;color:{color};">'
        f'{verdict.status.upper()}</span>'
        f'&nbsp;&nbsp;<span style="font-size:0.75em;color:{TEXT_M};">{verdict.what_looks_odd}</span>'
        f'</div>'
        f'<div>{live_badge} '
        f'<span style="color:{TEXT_D};font-size:0.68em;font-family:monospace;margin-left:8px;">'
        f'{latency_s:.2f}s</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── G4 capsule panel ─────────────────────────────────────────────────────────

def render_capsule_panel(
    packing: ArchivistPacking,
    hypotheses,
    available_evidence: list[dict],
    budget_bytes: int,
):
    """Side-by-side capsule: PARALLAX (G4 + knapsack) vs naive baseline."""
    hypo_names = [h.name for h in hypotheses] if hypotheses else []

    # Build EvidenceAssessment list from G4 scores.
    assessments = []
    for score in packing.scores:
        # Compress G4's 0–100 score down to the capsule's 1–5 priority scale.
        priority = max(1, min(5, round(score.score / 20)))
        assessments.append(EvidenceAssessment(
            evidence_id=score.evidence_id,
            priority=priority,
            reason=score.reason,
            supports_hypotheses=score.supports_hypotheses or hypo_names[:1],
            consequence_if_lost=f"Loses ground-side coverage of {', '.join(score.supports_hypotheses or ['primary hypothesis'])}",
        ))

    parallax_sel, parallax_rej = build_capsule(available_evidence, assessments, budget_bytes)
    naive_sel, naive_rej = naive_capsule(available_evidence, budget_bytes)

    p_used = sum(e["size_bytes"] for e in parallax_sel)
    n_used = sum(e["size_bytes"] for e in naive_sel)

    p_hyp = set()
    for e in parallax_sel:
        for a in assessments:
            if a.evidence_id == e["id"]:
                p_hyp.update(a.supports_hypotheses)
    # Naive baseline gets credit for supports it happens to include.
    n_hyp = set()
    for e in naive_sel:
        for a in assessments:
            if a.evidence_id == e["id"]:
                n_hyp.update(a.supports_hypotheses)

    left, right = st.columns(2)
    with left:
        _render_capsule_side("PARALLAX · G4 + KNAPSACK", parallax_sel, parallax_rej,
                              p_used, budget_bytes, p_hyp, GREEN, assessments)
    with right:
        _render_capsule_side("BASELINE · smallest-first", naive_sel, naive_rej,
                              n_used, budget_bytes, n_hyp, TEXT_D, assessments)

    st.markdown(
        f'<div style="margin-top:10px;padding:10px 14px;background:{NAVY}0c;border:1px solid {NAVY}55;'
        f'border-radius:8px;font-size:0.78em;color:{TEXT_M};">'
        f'Under a <strong>{budget_bytes // 1024} kB</strong> budget, PARALLAX preserves '
        f'<strong>{len(p_hyp)}</strong> of the surviving hypotheses vs. '
        f'<strong>{len(n_hyp)}</strong> for the naive baseline. Same byte budget, different answers home.'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_capsule_side(title, selected, rejected, used_bytes, budget, hyp_covered,
                          color, assessments):
    st.markdown(
        f'<div style="font-weight:800;color:{color};font-size:0.82em;'
        f'letter-spacing:0.05em;margin-bottom:6px;">{title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.72em;color:{TEXT_M};margin-bottom:8px;">'
        f'{used_bytes:,} / {budget:,} bytes used · '
        f'<strong style="color:{color};">{len(hyp_covered)} hypothesis path(s) preserved</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for item in selected:
        st.markdown(_card(
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span style="font-weight:600;font-size:0.75em;color:{TEXT};">{item["id"]}</span>'
            f'<span style="color:{TEXT_D};font-size:0.7em;font-family:monospace;">'
            f'{item["size_bytes"]:,} B</span>'
            f'</div>'
            f'<div style="font-size:0.7em;color:{TEXT_D};margin-top:2px;">{item.get("description","")}</div>',
            left_color=color,
        ), unsafe_allow_html=True)
    if rejected:
        rejected_names = ", ".join(e["id"] for e in rejected)
        st.markdown(
            f'<div style="font-size:0.68em;color:{TEXT_D};margin-top:4px;">'
            f'dropped: {rejected_names}</div>',
            unsafe_allow_html=True,
        )


# ── Benchmark table ─────────────────────────────────────────────────────────

def render_benchmark_report(report):
    if not report.outcomes:
        return

    rows = ""
    for o in report.outcomes:
        conflict_badge = _pill("CONFLICT", RED) if o.is_conflict else ""
        win = o.stack_score > o.fdir_only_score
        result_c = GREEN if win else (AMBER if o.stack_score == o.fdir_only_score else RED)
        veto_bits = []
        if o.validator_rejected: veto_bits.append("gate")
        if o.adjudicator_vetoed: veto_bits.append("G3")
        veto = " + ".join(veto_bits) or "—"

        rows += (
            f'<tr>'
            f'<td style="padding:6px 10px;font-family:monospace;color:{TEXT};font-size:0.72em;">{o.id}</td>'
            f'<td style="padding:6px 10px;color:{TEXT};font-size:0.75em;">{o.label} {conflict_badge}</td>'
            f'<td style="padding:6px 10px;text-align:center;color:{TEXT_M};font-family:monospace;font-size:0.72em;">{o.fdir_only_score}</td>'
            f'<td style="padding:6px 10px;text-align:center;color:{result_c};font-family:monospace;font-size:0.72em;font-weight:700;">{o.stack_score}</td>'
            f'<td style="padding:6px 10px;text-align:center;font-size:0.72em;">{veto}</td>'
            f'<td style="padding:6px 10px;text-align:center;color:{TEXT_M};font-family:monospace;font-size:0.72em;">{o.stack_plan_iterations}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{TEXT_M};font-family:monospace;font-size:0.72em;">{o.wall_time_s:.2f}s</td>'
            f'</tr>'
        )

    conf_ratio = f"{report.conflicts_caught_by_stack}/{report.total_conflicts}" if report.total_conflicts else "0/0"
    st.markdown(
        f'<div style="background:{WHITE};border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 14px;margin-bottom:10px;display:flex;gap:16px;flex-wrap:wrap;">'
        f'<div><div style="font-size:0.65em;color:{TEXT_D};font-weight:800;letter-spacing:0.09em;">STACK WINS</div>'
        f'<div style="font-size:1.2em;font-weight:800;color:{GREEN};">{report.stack_wins}/{report.n}</div></div>'
        f'<div><div style="font-size:0.65em;color:{TEXT_D};font-weight:800;letter-spacing:0.09em;">CONFLICTS CAUGHT</div>'
        f'<div style="font-size:1.2em;font-weight:800;color:{AMBER};">{conf_ratio}</div></div>'
        f'<div><div style="font-size:0.65em;color:{TEXT_D};font-weight:800;letter-spacing:0.09em;">UNSAFE PLANS BLOCKED</div>'
        f'<div style="font-size:1.2em;font-weight:800;color:{RED};">{report.unsafe_plans_blocked}</div></div>'
        f'<div><div style="font-size:0.65em;color:{TEXT_D};font-weight:800;letter-spacing:0.09em;">MEAN WALL TIME</div>'
        f'<div style="font-size:1.2em;font-weight:800;color:{NAVY};">{report.mean_wall_time_s:.2f}s</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="background:{WHITE};border:1px solid {BORDER};border-radius:10px;overflow:hidden;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead style="background:#f8fafc;">'
        f'<tr style="text-align:left;">'
        f'<th style="padding:8px 10px;font-size:0.64em;color:{TEXT_D};letter-spacing:0.09em;font-weight:800;">ID</th>'
        f'<th style="padding:8px 10px;font-size:0.64em;color:{TEXT_D};letter-spacing:0.09em;font-weight:800;">SCENARIO</th>'
        f'<th style="padding:8px 10px;font-size:0.64em;color:{TEXT_D};letter-spacing:0.09em;font-weight:800;text-align:center;">FDIR</th>'
        f'<th style="padding:8px 10px;font-size:0.64em;color:{TEXT_D};letter-spacing:0.09em;font-weight:800;text-align:center;">STACK</th>'
        f'<th style="padding:8px 10px;font-size:0.64em;color:{TEXT_D};letter-spacing:0.09em;font-weight:800;text-align:center;">VETO</th>'
        f'<th style="padding:8px 10px;font-size:0.64em;color:{TEXT_D};letter-spacing:0.09em;font-weight:800;text-align:center;">ITER</th>'
        f'<th style="padding:8px 10px;font-size:0.64em;color:{TEXT_D};letter-spacing:0.09em;font-weight:800;text-align:right;">WALL</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="color:{TEXT_D};font-size:0.7em;margin-top:8px;line-height:1.55;">'
        f'Score is a subsystem health metric: nominal=3, recovering=2, degraded=1, failed=0. '
        f'Stack column reflects post-plan projected state when the plan was approved by both '
        f'the deterministic gate and G3; otherwise it falls back to the FDIR-only outcome. '
        f'Wall column is stack latency, dwarfed by the 38-minute Earth round trip.'
        f'</div>',
        unsafe_allow_html=True,
    )
