import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import html
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parallax.spacecraft import (
    FAULT_DEFINITIONS,
    VISUAL_COMPONENTS,
    SpacecraftState,
    add_noise,
    copy_state,
    fault_details,
    inject_custom_fault,
    inject_fault,
)
from parallax.fdir import run_fdir, FDIRReport
from parallax.gemma import run_predictive_analysis
from parallax.models import GemmaPredictiveAnalysis
from parallax.satellite_view import STATUS_COLORS, build_satellite_figure, component_states
from parallax.utils import format_bytes

st.set_page_config(page_title="PARALLAX FDIR", page_icon="🛸", layout="wide", initial_sidebar_state="collapsed")

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
    .subsystem-card { background:#0d1117; border:1px solid #1e293b; border-radius:8px; padding:12px 14px; margin:4px 0; }
    .subsystem-nominal { border-left:4px solid #22c55e; }
    .subsystem-degraded { border-left:4px solid #f59e0b; }
    .subsystem-failed { border-left:4px solid #ef4444; }
    .subsystem-recovering { border-left:4px solid #3b82f6; }
    .fdir-log { font-family:monospace; font-size:0.82em; background:#0d1117; border:1px solid #1e293b; border-radius:6px; padding:12px; max-height:350px; overflow-y:auto; }
    .fdir-active { background:#1a0a2e; border:1px solid #7c3aed; border-radius:6px; padding:8px 14px; margin:6px 0; }
    .fdir-monitoring { background:#0f1a0f; border:1px solid #16a34a; border-radius:6px; padding:8px 14px; margin:6px 0; }
    .prediction-card { background:#111827; border-left:3px solid #f59e0b; border-radius:4px; padding:10px 14px; margin:6px 0; }
    .fault-btn-bar { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }
    .sensor-reading { font-family:monospace; font-size:0.88em; color:#94a3b8; }
    .sensor-alert { color:#f87171; font-weight:bold; }
    .sensor-nominal { color:#94a3b8; }
</style>
""", unsafe_allow_html=True)


def health_color(status: str) -> str:
    return {"nominal": "#22c55e", "degraded": "#f59e0b", "failed": "#ef4444", "recovering": "#3b82f6"}.get(status, "#64748b")

def health_icon(status: str) -> str:
    return {"nominal": "●", "degraded": "◑", "failed": "○", "recovering": "↻"}.get(status, "?")


def render_subsystem_card(name: str, status: str, readings: list) -> None:
    """readings: list of (label, value, is_alert)"""
    color = health_color(status)
    icon = health_icon(status)
    cls = f"subsystem-card subsystem-{status}"
    rows = "".join(
        f"<div class='sensor-reading'><span style='color:#475569;'>├ {label}:</span> "
        f"<span class=\"{'sensor-alert' if alert else 'sensor-nominal'}\">{value}</span></div>"
        for label, value, alert in readings
    )
    st.markdown(f"""
    <div class="{cls}">
      <div style="margin-bottom:6px;">
        <span style="color:{color}; font-size:1.1em;">{icon}</span>
        <strong style="color:#e0e8ff; margin-left:6px;">{name}</strong>
        <span style="float:right; color:{color}; font-size:0.78em; font-weight:bold;">{status.upper()}</span>
      </div>
      {rows}
    </div>
    """, unsafe_allow_html=True)


def render_fdir_log(report: FDIRReport) -> None:
    phase_colors = {"detection": "#ef4444", "isolation": "#f59e0b", "recovery": "#60a5fa", "monitoring": "#22c55e"}
    outcome_icons = {"success": "✓", "partial": "◑", "in_progress": "…", "failed": "✗"}

    lines = []
    for action in report.actions:
        color = phase_colors.get(action.phase, "#94a3b8")
        icon = outcome_icons.get(action.outcome, "")
        lines.append(
            f'<div style="margin:3px 0;">'
            f'<span style="color:#475569;">[{action.timestamp}]</span> '
            f'<span style="color:{color};">{icon} {html.escape(action.message)}</span>'
            f'</div>'
        )

    content = "\n".join(lines) or '<span style="color:#475569;">No events recorded.</span>'
    st.markdown(f'<div class="fdir-log">{content}</div>', unsafe_allow_html=True)


def stability_color(s: str) -> str:
    return {"stable": "#22c55e", "degraded": "#f59e0b", "critical": "#ef4444", "unknown": "#64748b"}.get(s, "#64748b")


def activate_fault(state: SpacecraftState, fault_id: str) -> None:
    """Run deterministic FDIR then ask Gemma to reason over this exact event."""
    st.session_state.spacecraft = state
    fdir_report = run_fdir(state)
    st.session_state.fdir_report = fdir_report
    st.session_state.active_fault_id = fault_id

    state_dict = {k: v for k, v in state.__dict__.items() if not k.startswith("_")}
    fdir_dict = {
        "triggered": fdir_report.triggered,
        "active_faults": fdir_report.active_faults,
        "actions": [{"timestamp": a.timestamp, "phase": a.phase, "message": a.message, "outcome": a.outcome} for a in fdir_report.actions],
        "mission_safe": fdir_report.mission_safe,
        "summary": fdir_report.summary,
    }
    mission_context = "Asteria-7, Jupiter approach cruise, Earth delay 38 minutes, transmission budget 25 KB, autonomous operations required."
    fault_context = state.fault_metadata.get(fault_id)
    # A preset added after a custom event still needs the live custom context;
    # do not fall back to a stale canned prediction for the preset alone.
    if fault_context is None and state.fault_metadata:
        active_custom = list(state.fault_metadata.values())
        fault_context = {
            "label": fault_details(state, fault_id)["label"],
            "description": "Active operator-defined events are present alongside this FDIR event.",
            "subsystems": sorted({subsystem for event in active_custom for subsystem in event.get("subsystems", [])}),
            "severity": max((event.get("severity", "warning") for event in active_custom), key={"advisory": 0, "warning": 1, "critical": 2}.get),
            "active_operator_events": active_custom,
        }

    try:
        prediction, is_live = run_predictive_analysis(
            state_dict, fdir_dict, mission_context, fault_id, fault_context=fault_context,
        )
        st.session_state.prediction = prediction
        st.session_state.prediction_is_live = is_live
        st.session_state.prediction_source = (
            "Live Gemma" if is_live else ("Dynamic local fallback" if fault_context else "Cached Gemma analysis")
        )
    except Exception as exc:
        st.error(f"Gemma prediction failed: {exc}")


def main():
    # Header
    st.markdown('<h1 style="color:#60a5fa; margin-bottom:0; font-size:2em; letter-spacing:0.06em;">PARALLAX FDIR</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; margin-top:0;">Autonomous fault detection, isolation, and recovery — with Gemma predictive intelligence</p>', unsafe_allow_html=True)

    # Session state init
    if "spacecraft" not in st.session_state:
        st.session_state.spacecraft = SpacecraftState()
        st.session_state.fdir_report = None
        st.session_state.prediction = None
        st.session_state.prediction_is_live = False
        st.session_state.prediction_source = None
        st.session_state.active_fault_id = None
        st.session_state.tick = 0

    # Advance simulation tick for noise
    st.session_state.tick += 1
    state: SpacecraftState = st.session_state.spacecraft
    noisy_state = add_noise(copy_state(state), seed=st.session_state.tick)

    # Mission status bar
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        st.metric("Spacecraft", "Asteria-7")
    with col_b:
        st.metric("Mission Phase", "Jupiter Approach")
    with col_c:
        st.metric("Earth Delay", "38 min")
    with col_d:
        healthy = sum(1 for v in state.subsystem_health.values() if v == "nominal")
        st.metric("Systems Nominal", f"{healthy} / 6")
    with col_e:
        overall = "NOMINAL" if healthy == 6 else ("DEGRADED" if healthy >= 4 else "CRITICAL")
        color = "#22c55e" if overall == "NOMINAL" else ("#f59e0b" if overall == "DEGRADED" else "#ef4444")
        st.markdown(f'<div style="background:{color}22; border:1px solid {color}; border-radius:6px; padding:8px 14px; text-align:center; margin-top:4px;"><strong style="color:{color};">{overall}</strong></div>', unsafe_allow_html=True)

    st.markdown("---")

    # The composer provides an open-ended event path.  Telemetry overrides are
    # deliberately explicit so model-generated prose never changes spacecraft state.
    st.markdown("### Event Composer")
    st.caption("Define any observed anomaly, select the affected hardware, and optionally apply validated telemetry changes. Gemma receives the full event context rather than a fixed fault label.")
    with st.form("custom_event_form", clear_on_submit=False):
        event_col, impact_col = st.columns([1.25, 1])
        with event_col:
            custom_title = st.text_input("Event title", placeholder="e.g. Propellant-line pressure oscillation")
            custom_description = st.text_area(
                "Observed condition for Gemma",
                placeholder="Describe what was observed, when it began, and any uncertainty. This is included verbatim in the analysis context.",
                height=92,
            )
        with impact_col:
            custom_severity = st.select_slider("Severity", options=["advisory", "warning", "critical"], value="warning")
            custom_subsystems = st.multiselect("Affected subsystems", list(state.subsystem_health), placeholder="Select one or more")
            custom_components = st.multiselect("Highlight in 3D model", VISUAL_COMPONENTS, placeholder="Optional component-level status")
        with st.expander("Optional telemetry overrides (JSON)"):
            custom_overrides = st.text_area(
                "Validated state changes",
                value="{}",
                help="Example: {\"tank_pressure_bar\": 165, \"thruster_status\": \"disabled\"}. Allowed fields include bus_voltage_v, pcu_temp_c, solar_output_w, signal_strength_dbm, tank_pressure_bar, thruster_status, antenna_mode, and spectrometer_status.",
                height=80,
            )
        submit_custom = st.form_submit_button("Analyse custom event", type="primary", use_container_width=True)

    if submit_custom:
        try:
            overrides = json.loads(custom_overrides or "{}")
            new_state, custom_id = inject_custom_fault(
                copy_state(state), title=custom_title, description=custom_description,
                subsystems=custom_subsystems, severity=custom_severity,
                telemetry_overrides=overrides, components=custom_components,
            )
            activate_fault(new_state, custom_id)
            st.rerun()
        except json.JSONDecodeError as exc:
            st.error(f"Telemetry overrides must be valid JSON: {exc.msg}")
        except ValueError as exc:
            st.error(str(exc))

    with st.expander("Quick fault examples", expanded=False):
        st.caption("Optional presets for a fast demo. Custom events above are not limited to this set.")
        fault_cols = st.columns(3)
        for index, (fault_id, fault_def) in enumerate(FAULT_DEFINITIONS.items()):
            with fault_cols[index % 3]:
                active = fault_id in state.active_faults
                label = f"{fault_def['icon']} {fault_def['label']}"
                if active:
                    st.markdown(f'<div style="background:#7f1d1d22; border:1px solid #ef4444; border-radius:6px; padding:6px; text-align:center; font-size:0.8em; color:#ef4444;">⚠ {html.escape(fault_def["label"])}</div>', unsafe_allow_html=True)
                elif st.button(label, key=f"fault_{fault_id}", use_container_width=True):
                    new_state = inject_fault(copy_state(state), fault_id)
                    activate_fault(new_state, fault_id)
                    st.rerun()

    # Reset button
    if state.active_faults:
        if st.button("Reset to Nominal", type="secondary"):
            st.session_state.spacecraft = SpacecraftState()
            st.session_state.fdir_report = None
            st.session_state.prediction = None
            st.session_state.prediction_source = None
            st.session_state.active_fault_id = None
            st.rerun()

    st.markdown("---")

    viewer_col, component_col = st.columns([1.55, 1])
    with viewer_col:
        st.markdown("### Live 3D Spacecraft State")
        st.caption("Drag to inspect. Colours and hover details are derived from the same live state as the telemetry cards.")
        st.plotly_chart(build_satellite_figure(noisy_state), width="stretch", config={"displayModeBar": False})
    with component_col:
        st.markdown("### Hardware State")
        issues = [item for item in component_states(noisy_state) if item["status"] not in {"nominal", "standby"}]
        if not issues:
            st.markdown('<div class="fdir-monitoring"><strong style="color:#22c55e;">● ALL MODELLED HARDWARE NOMINAL</strong><br><span style="color:#64748b; font-size:0.88em;">Camera and low-gain antenna are in standby where expected.</span></div>', unsafe_allow_html=True)
        else:
            for item in issues:
                component_color = STATUS_COLORS.get(item["status"], "#64748b")
                st.markdown(
                    f'<div class="subsystem-card" style="border-left-color:{component_color};">'
                    f'<strong style="color:{component_color};">{html.escape(item["name"])}</strong>'
                    f'<span style="float:right; color:{component_color}; font-size:0.78em;">{item["status"].upper()}</span><br>'
                    f'<span style="color:#94a3b8; font-size:0.84em;">{html.escape(item["detail"])}</span></div>',
                    unsafe_allow_html=True,
                )

    fdir_report: FDIRReport = st.session_state.fdir_report
    prediction: GemmaPredictiveAnalysis = st.session_state.prediction

    # Three-column main layout
    left_col, center_col, right_col = st.columns([1.1, 1.2, 1.1])

    with left_col:
        st.markdown("### Spacecraft Systems")

        # Power
        power_status = noisy_state.subsystem_health.get("Power", "nominal")
        render_subsystem_card("Power", power_status, [
            ("Solar Output", f"{noisy_state.solar_output_w:.0f} W ({noisy_state.solar_efficiency_pct:.0f}%)", noisy_state.solar_output_w < 650),
            ("Battery SoC", f"{noisy_state.battery_soc_pct:.1f}%", noisy_state.battery_soc_pct < 80),
            ("PCU Temp", f"{noisy_state.pcu_temp_c:.1f}°C", noisy_state.pcu_temp_c > 50),
            ("Bus Voltage", f"{noisy_state.bus_voltage_v:.2f} V", noisy_state.bus_voltage_v < 27.0),
        ])

        # Thermal
        thermal_status = noisy_state.subsystem_health.get("Thermal", "nominal")
        render_subsystem_card("Thermal", thermal_status, [
            ("Bus Temp", f"{noisy_state.bus_temp_c:.1f}°C", noisy_state.bus_temp_c > 34),
            ("Instrument Temp", f"{noisy_state.instrument_temp_c:.1f}°C", noisy_state.instrument_temp_c > 26),
            ("Radiator", "ACTIVE" if noisy_state.radiator_active else "STANDBY", False),
        ])

        # ADCS
        adcs_status = noisy_state.subsystem_health.get("ADCS", "nominal")
        render_subsystem_card("ADCS", adcs_status, [
            ("Attitude Error", f"{noisy_state.attitude_error_arcsec:.2f} arcsec", noisy_state.attitude_error_arcsec > 5),
            ("RW Speed", f"{noisy_state.rw_speed_rpm:.0f} rpm", noisy_state.rw_speed_rpm < 600 or noisy_state.rw_speed_rpm > 4500),
            ("Sun Error", f"{noisy_state.sun_pointing_error_deg:.3f}°", noisy_state.sun_pointing_error_deg > 0.3),
            ("Mode", noisy_state.adcs_mode.replace("_", " ").upper(), noisy_state.adcs_mode != "fine_pointing"),
        ])

        # Comms
        comms_status = noisy_state.subsystem_health.get("Communications", "nominal")
        render_subsystem_card("Communications", comms_status, [
            ("Signal", f"{noisy_state.signal_strength_dbm:.1f} dBm", noisy_state.signal_strength_dbm < -121),
            ("Link Margin", f"{noisy_state.link_margin_db:.1f} dB", noisy_state.link_margin_db < 3),
            ("Data Rate", f"{noisy_state.data_rate_mbps:.3f} Mbps", noisy_state.data_rate_mbps < 0.2),
            ("Antenna", noisy_state.antenna_mode.upper().replace("_", " "), noisy_state.antenna_mode != "hga"),
        ])

        # Science
        science_status = noisy_state.subsystem_health.get("Science", "nominal")
        spec_val = f"{noisy_state.spectrometer_output:.1f} cts" if noisy_state.spectrometer_output is not None else "OFFLINE"
        render_subsystem_card("Science Instruments", science_status, [
            ("Spectrometer", noisy_state.spectrometer_status.upper(), noisy_state.spectrometer_status == "failed"),
            ("Output", spec_val, noisy_state.spectrometer_output is None),
            ("Camera", noisy_state.camera_status.upper(), False),
            ("Radiation", f"{noisy_state.radiation_cps:.1f} cps", False),
        ])

        # Propulsion
        prop_status = noisy_state.subsystem_health.get("Propulsion", "nominal")
        render_subsystem_card("Propulsion", prop_status, [
            ("Tank Pressure", f"{noisy_state.tank_pressure_bar:.1f} bar", noisy_state.tank_pressure_bar < 180),
            ("Fuel Mass", f"{noisy_state.fuel_mass_kg:.1f} kg", noisy_state.fuel_mass_kg < 10),
            ("Thrusters", noisy_state.thruster_status.upper(), noisy_state.thruster_status != "nominal"),
        ])

    with center_col:
        st.markdown("### FDIR Console")

        if fdir_report is None or not fdir_report.triggered:
            st.markdown('<div class="fdir-monitoring"><strong style="color:#22c55e;">● FDIR STATUS: MONITORING</strong><br><span style="color:#64748b; font-size:0.88em;">All systems within nominal parameters. No faults detected.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fdir-active"><strong style="color:#ef4444;">⚡ FDIR STATUS: {"SAFE MODE" if fdir_report.mission_safe else "CRITICAL"}</strong><br><span style="color:#94a3b8; font-size:0.88em;">{html.escape(fdir_report.summary)}</span></div>', unsafe_allow_html=True)

            st.markdown("**Active Faults:**")
            for fault in fdir_report.active_faults:
                fault_def = fault_details(state, fault)
                st.markdown(f'<span style="background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:4px; font-size:0.82em; margin:2px;">{fault_def.get("icon","⚠")} {html.escape(fault_def.get("label", fault))}</span>', unsafe_allow_html=True)

        st.markdown("**Recovery Log:**")
        if fdir_report:
            render_fdir_log(fdir_report)
        else:
            st.markdown('<div class="fdir-log"><span style="color:#475569;">[T+00:00] System initialised. FDIR monitoring active.<br>[T+00:00] All subsystems nominal.</span></div>', unsafe_allow_html=True)

        # Telemetry sparkline charts for key sensors
        if fdir_report and fdir_report.triggered:
            st.markdown("**Sensor Timeline (simulated window):**")
            rng = np.random.default_rng(42)
            t = np.arange(0, 60)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=["Solar Output (W)", "Battery SoC (%)"])

            # Before fault: nominal; after fault: degraded
            solar_pre = rng.normal(850, 4, 40)
            solar_post = rng.normal(noisy_state.solar_output_w, 4, 20)
            solar_series = np.concatenate([solar_pre, solar_post])

            bat_pre = rng.normal(94.5, 0.05, 40)
            bat_post = np.linspace(94.2, noisy_state.battery_soc_pct, 20) + rng.normal(0, 0.05, 20)
            bat_series = np.concatenate([bat_pre, bat_post])

            fig.add_trace(go.Scatter(x=t, y=solar_series, mode="lines", line=dict(color="#fbbf24", width=1.5), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=bat_series, mode="lines", line=dict(color="#34d399", width=1.5), showlegend=False), row=2, col=1)
            fig.add_vline(x=40, line_dash="dash", line_color="#ef4444", line_width=1.5)
            fig.update_layout(height=200, plot_bgcolor="#0d1117", paper_bgcolor="#0a0e1a",
                              font=dict(color="#94a3b8", size=10), margin=dict(l=5, r=5, t=25, b=5))
            fig.update_xaxes(gridcolor="#1e293b")
            fig.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig, width="stretch")

    with right_col:
        st.markdown("### Gemma Intelligence")

        if prediction is None:
            st.markdown('<div class="metric-card" style="text-align:center; padding:30px;"><span style="color:#475569;">Compose any event or choose a quick example to activate<br>Gemma predictive analysis</span></div>', unsafe_allow_html=True)
        else:
            if not st.session_state.prediction_is_live:
                st.info(f"Source: {st.session_state.prediction_source or 'Cached Gemma analysis'}.")

            stab_color = stability_color(prediction.system_stability)
            st.markdown(f'<div style="background:{stab_color}22; border:1px solid {stab_color}; border-radius:6px; padding:8px 14px; margin-bottom:8px;"><strong style="color:{stab_color};">SYSTEM STABILITY: {prediction.system_stability.upper()}</strong></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="metric-card"><strong style="color:#94a3b8;">Assessment</strong><br><span style="color:#cbd5e1; font-size:0.88em;">{html.escape(prediction.current_assessment)}</span></div>', unsafe_allow_html=True)

            if prediction.predicted_failures:
                st.markdown("**Predicted Next Failures:**")
                for pf in prediction.predicted_failures:
                    prob_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(pf.probability, "#94a3b8")
                    st.markdown(f"""
                    <div class="prediction-card">
                      <span style="color:{prob_color}; font-weight:bold;">⚠ {html.escape(pf.subsystem)}</span>
                      <span style="color:{prob_color}; font-size:0.78em; float:right;">{pf.probability.upper()}</span><br>
                      <span style="color:#94a3b8; font-size:0.85em;">{html.escape(pf.failure_mode)}</span><br>
                      <span style="color:#64748b; font-size:0.8em;">⏱ {html.escape(pf.estimated_time_to_failure)}</span>
                    </div>
                    """, unsafe_allow_html=True)

            if prediction.cascading_risks:
                with st.expander("Cascading Risks"):
                    for risk in prediction.cascading_risks:
                        st.markdown(f"- {risk}")

            st.markdown("**Recommended Actions:**")
            for i, action in enumerate(prediction.recommended_actions, 1):
                st.markdown(f'<div style="background:#0f1629; border-left:2px solid #3b82f6; padding:6px 10px; margin:3px 0; border-radius:3px; font-size:0.87em; color:#cbd5e1;">{i}. {html.escape(action)}</div>', unsafe_allow_html=True)

    # Earth Report section (full width)
    if prediction:
        st.markdown("---")
        st.markdown("### Earth Transmission Report")

        col_report, col_meta = st.columns([3, 1])
        with col_report:
            st.code(prediction.earth_report, language=None)
        with col_meta:
            fault_id = st.session_state.active_fault_id or "unknown"
            report_payload = {
                "mission": "Asteria-7",
                "fault_id": fault_id,
                "fdir_summary": fdir_report.summary if fdir_report else "",
                "fdir_actions": [{"timestamp": a.timestamp, "message": a.message, "outcome": a.outcome} for a in (fdir_report.actions if fdir_report else [])],
                "gemma_assessment": prediction.current_assessment,
                "predicted_failures": [pf.model_dump() for pf in prediction.predicted_failures],
                "recommended_actions": prediction.recommended_actions,
                "system_stability": prediction.system_stability,
                "earth_report": prediction.earth_report,
            }
            report_bytes = json.dumps(report_payload, indent=2).encode()
            st.metric("Report Size", format_bytes(len(report_bytes)))
            st.metric("Confidence", prediction.confidence.upper())
            st.metric("Source", st.session_state.prediction_source or ("Live Gemma" if st.session_state.prediction_is_live else "Cached"))
            st.download_button(
                "Download Earth Report",
                data=json.dumps(report_payload, indent=2),
                file_name=f"parallax_earth_report_{fault_id}.json",
                mime="application/json",
            )

    st.markdown("---")
    st.markdown('<p style="text-align:center; color:#334155; font-size:0.85em;">PARALLAX FDIR — Autonomous fault detection, isolation, recovery, and predictive intelligence for deep-space missions</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
