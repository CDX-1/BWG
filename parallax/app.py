import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from parallax.utils import load_scenario, format_bytes
from parallax.simulator import generate_spectrometer_scenario, ANOMALY_TIME_INDEX, MISSION_TIME_LABEL
from parallax.detector import detect_event
from parallax.retrieval import load_knowledge_chunks, retrieve
from parallax.gemma import run_parallax_analysis
from parallax.capsule import build_capsule, hypotheses_covered
from parallax.baseline import naive_capsule
from parallax.models import ParallaxAnalysis

st.set_page_config(
    page_title="PARALLAX",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark space theme
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; }
    .main .block-container { padding-top: 1rem; max-width: 100%; }
    h1, h2, h3 { color: #e0e8ff; }
    .metric-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .hypothesis-card {
        background: #111827;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .alert-badge {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: bold;
    }
    .nominal-badge {
        background: #14532d;
        color: #86efac;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
    }
    .selected-evidence {
        background: #0f2a1f;
        border: 1px solid #16a34a;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    .rejected-evidence {
        background: #1a0a0a;
        border: 1px solid #7f1d1d;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        opacity: 0.7;
    }
    .step-indicator {
        display: inline-block;
        background: #1e3a5f;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.8em;
    }
    .step-done {
        background: #14532d;
        color: #86efac;
    }
</style>
""", unsafe_allow_html=True)


def confidence_color(c: str) -> str:
    return {"low": "#ef4444", "medium": "#f59e0b", "high": "#22c55e"}.get(c, "#94a3b8")


def render_telemetry_charts(df: pd.DataFrame, anomaly_idx: int) -> None:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=["Spectrometer Intensity", "Radiation Count", "Instrument Temperature (°C)", "Checksum Errors"],
        vertical_spacing=0.06,
    )

    anom_time = df["time"][anomaly_idx]

    for row, (col, color, unit) in enumerate([
        ("spectrum", "#60a5fa", "counts"),
        ("radiation", "#f87171", "counts/s"),
        ("temperature", "#34d399", "°C"),
        ("checksum_errors", "#fbbf24", "errors"),
    ], 1):
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[col],
            mode="lines",
            line=dict(color=color, width=1.5),
            name=col,
            showlegend=False,
        ), row=row, col=1)

        # Anomaly marker
        fig.add_vline(
            x=anom_time,
            line_dash="dash",
            line_color="#ff6b6b",
            line_width=1.5,
            row=row, col=1,
        )

    fig.update_layout(
        height=480,
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="#1e293b", title_text="Mission Elapsed Time (s)", row=4, col=1)
    fig.update_yaxes(gridcolor="#1e293b")

    st.plotly_chart(fig, width="stretch")


def render_hypothesis_card(h, idx: int) -> None:
    color = confidence_color(h.confidence)
    st.markdown(f"""
    <div class="hypothesis-card" style="border-left-color: {color};">
        <strong style="color: {color};">{idx}. {h.name}</strong>
        &nbsp;<span style="color:{color}; font-size:0.8em;">confidence: {h.confidence}</span>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details", expanded=False):
        if h.supporting_evidence:
            st.markdown(f"**Supports:** {', '.join(h.supporting_evidence)}")
        if h.contradicting_evidence:
            st.markdown(f"**Contradicts:** {', '.join(h.contradicting_evidence)}")
        if h.evidence_needed:
            st.markdown(f"**Needs:** {', '.join(h.evidence_needed)}")


def render_evidence_item(item: dict, assessment=None, selected: bool = True) -> None:
    cls = "selected-evidence" if selected else "rejected-evidence"
    icon = "✓" if selected else "✗"
    color = "#22c55e" if selected else "#ef4444"
    reason_text = f"<br><small style='color:#94a3b8;'>{assessment.reason[:120]}...</small>" if assessment and assessment.reason else ""
    priority_text = f"<span style='color:#fbbf24; margin-left:8px;'>P{assessment.priority}</span>" if assessment else ""
    st.markdown(f"""
    <div class="{cls}">
        <span style="color:{color}; font-weight:bold;">{icon}</span>
        <strong style="color:#e0e8ff;"> {item['id']}</strong>
        {priority_text}
        <span style="color:#64748b; float:right;">{format_bytes(item['size_bytes'])}</span>
        {reason_text}
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <h1 style="color:#60a5fa; margin-bottom:0; font-size:2.2em; letter-spacing:0.08em;">PARALLAX</h1>
    <p style="color:#64748b; margin-top:0; font-size:0.95em;">Evidence-preserving autonomy for delayed deep-space missions</p>
    """, unsafe_allow_html=True)

    # Load scenario
    scenario = load_scenario("spectrometer_001")
    mission = scenario["mission"]
    budget = mission["transmission_budget_bytes"]

    # Mission status bar
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Spacecraft", mission["spacecraft"])
    with col_b:
        st.metric("Mission Phase", "Europa Flyby")
    with col_c:
        st.metric("Earth Delay", f"{mission['earth_delay_minutes']} min")
    with col_d:
        st.metric("TX Budget", format_bytes(budget))

    st.markdown("---")

    # Run analysis button
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.analysis = None
        st.session_state.is_live = False
        st.session_state.retrieved = []
        st.session_state.step = 0

    if not st.session_state.analysis_done:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("Run PARALLAX Analysis", type="primary")
        with col_info:
            st.markdown("<p style='color:#64748b; padding-top:8px;'>Simulate anomaly detection, retrieve context, and generate evidence capsule.</p>", unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Running PARALLAX pipeline..."):
                # Step 1: Simulate
                df = generate_spectrometer_scenario()
                triggered, det_details = detect_event(df)
                st.session_state.df = df
                st.session_state.det_details = det_details
                st.session_state.step = 1

                # Step 2: Retrieve
                chunks = load_knowledge_chunks()
                query = f"{scenario['event']['description']} {scenario['event']['trigger']} spectrometer radiation checksum"
                retrieved = retrieve(query, chunks, top_k=5)
                st.session_state.retrieved = retrieved
                st.session_state.step = 2

                # Step 3: Reason
                analysis, is_live = run_parallax_analysis(scenario, retrieved)
                st.session_state.analysis = analysis
                st.session_state.is_live = is_live
                st.session_state.step = 3

                st.session_state.analysis_done = True
                st.rerun()
    else:
        # Show reset button
        if st.button("Reset"):
            for key in ["analysis_done", "analysis", "is_live", "retrieved", "step", "df", "det_details"]:
                st.session_state.pop(key, None)
            st.rerun()

    if not st.session_state.analysis_done:
        # Show placeholder telemetry
        df = generate_spectrometer_scenario()
        render_telemetry_charts(df, ANOMALY_TIME_INDEX)
        return

    # Unpack state
    df = st.session_state.df
    det_details = st.session_state.det_details
    retrieved = st.session_state.retrieved
    analysis: ParallaxAnalysis = st.session_state.analysis
    is_live = st.session_state.is_live
    available_evidence = scenario["available_evidence"]

    # Fallback indicator
    if not is_live:
        st.warning("Demo fallback active: displaying a previously generated Gemma analysis.")

    # Progress steps
    steps = ["Detect", "Retrieve", "Reason", "Preserve", "Transmit"]
    step_html = " ".join(
        f'<span class="step-indicator step-done">✓ {s}</span>'
        for s in steps
    )
    st.markdown(step_html, unsafe_allow_html=True)

    # Resolution status banner
    status_colors = {
        "unresolved": "#7c3aed",
        "known_event": "#0284c7",
        "likely_fault": "#dc2626",
        "likely_scientific_event": "#16a34a",
    }
    status_color = status_colors.get(analysis.resolution_status, "#64748b")
    st.markdown(f"""
    <div style="background:{status_color}22; border:1px solid {status_color}; border-radius:6px; padding:10px 16px; margin:8px 0;">
        <strong style="color:{status_color};">STATUS: {analysis.resolution_status.upper().replace('_', ' ')}</strong>
        &nbsp;|&nbsp; <span style="color:#94a3b8;">{analysis.event_summary}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Main three-column layout
    left_col, mid_col, right_col = st.columns([1.1, 1.3, 1.1])

    with left_col:
        st.markdown("### Telemetry")
        det = det_details
        st.markdown(f"""
        <div class="metric-card">
            Event Time: <strong style="color:#fbbf24;">{MISSION_TIME_LABEL}</strong><br>
            Spectrum Peak: <span class="alert-badge">{det['spectrum_peak']:.1f}</span>
            &nbsp;Radiation Peak: <span class="alert-badge">{det['radiation_peak']:.1f}</span>
            &nbsp;Checksum Errors: <span class="alert-badge">{det['checksum_errors']}</span>
        </div>
        """, unsafe_allow_html=True)
        render_telemetry_charts(df, det_details["anomaly_index"])

    with mid_col:
        st.markdown("### Gemma Analysis")

        st.markdown("**Competing Hypotheses**")
        for i, h in enumerate(analysis.hypotheses, 1):
            render_hypothesis_card(h, i)

        st.markdown(f"""
        <div class="metric-card" style="margin-top:12px;">
            <strong style="color:#fbbf24;">Uncertainty</strong><br>
            <span style="color:#94a3b8; font-size:0.9em;">{analysis.uncertainty_statement}</span>
        </div>
        <div class="metric-card">
            <strong style="color:#60a5fa;">Compression Warning</strong><br>
            <span style="color:#94a3b8; font-size:0.9em;">{analysis.compression_warning}</span>
        </div>
        <div class="metric-card">
            <strong style="color:#34d399;">Recommended Follow-Up</strong><br>
            <span style="color:#94a3b8; font-size:0.9em;">{analysis.recommended_follow_up}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Retrieved Knowledge Passages"):
            for r in retrieved:
                st.markdown(f"**[{r['source']}]**")
                st.text(r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"])

    with right_col:
        st.markdown("### Evidence Capsule")

        assessment_map = {a.evidence_id: a for a in analysis.preservation_priorities}
        selected, rejected = build_capsule(available_evidence, analysis.preservation_priorities, budget)
        used_bytes = sum(e["size_bytes"] for e in selected)
        covered = hypotheses_covered(selected, analysis.preservation_priorities)

        # Budget progress bar
        pct = used_bytes / budget
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <span style="color:#94a3b8;">Budget used: </span>
            <strong style="color:#60a5fa;">{format_bytes(used_bytes)}</strong>
            <span style="color:#64748b;"> / {format_bytes(budget)}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(pct, 1.0))

        st.markdown(f"**Hypotheses still testable:** {len(covered)} / {len(analysis.hypotheses)}")
        if covered:
            st.markdown(", ".join(f"`{h}`" for h in covered))

        st.markdown("**Selected:**")
        for item in selected:
            render_evidence_item(item, assessment_map.get(item["id"]), selected=True)

        if rejected:
            st.markdown("**Rejected:**")
            for item in rejected:
                render_evidence_item(item, assessment_map.get(item["id"]), selected=False)

    # Bottom comparison panel
    st.markdown("---")
    st.markdown("### Baseline Comparison")

    naive_selected, naive_rejected = naive_capsule(available_evidence, budget)
    naive_covered_ids = {e["id"] for e in naive_selected}

    par_col, naive_col = st.columns(2)

    with par_col:
        st.markdown("#### PARALLAX")
        par_ids = {e["id"] for e in selected}
        used_p = sum(e["size_bytes"] for e in selected)
        st.markdown(f"*Preserves: {format_bytes(used_p)} — {len(covered)} hypothesis paths testable*")
        for item in available_evidence:
            is_sel = item["id"] in par_ids
            icon = "✓" if is_sel else "✗"
            color = "#22c55e" if is_sel else "#ef4444"
            reason = ""
            if assessment_map.get(item["id"]):
                a = assessment_map[item["id"]]
                reason = f" — {', '.join(a.supports_hypotheses[:2])}" if a.supports_hypotheses else ""
            st.markdown(f"<span style='color:{color};'>{icon}</span> **{item['id']}** ({format_bytes(item['size_bytes'])}){reason}", unsafe_allow_html=True)

    with naive_col:
        st.markdown("#### Naive Compression (smallest-first)")
        used_n = sum(e["size_bytes"] for e in naive_selected)

        # Work out which hypotheses are still testable with naive selection
        naive_hyp_covered = set()
        for h in analysis.hypotheses:
            for eid in h.supporting_evidence:
                if eid in naive_covered_ids:
                    naive_hyp_covered.add(h.name)

        st.markdown(f"*Preserves: {format_bytes(used_n)} — {len(naive_hyp_covered)} hypothesis paths testable*")
        for item in available_evidence:
            is_sel = item["id"] in naive_covered_ids
            icon = "✓" if is_sel else "✗"
            color = "#22c55e" if is_sel else "#ef4444"
            st.markdown(f"<span style='color:{color};'>{icon}</span> **{item['id']}** ({format_bytes(item['size_bytes'])})", unsafe_allow_html=True)

        lost = set()
        for h in analysis.hypotheses:
            if h.name not in naive_hyp_covered:
                lost.add(h.name)
        if lost:
            st.markdown(f"<span style='color:#ef4444;'>Lost investigative paths: {', '.join(lost)}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#334155; font-size:0.85em;'>Existing systems transmit what they understand. PARALLAX preserves the evidence they do not.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
