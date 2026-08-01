import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time

import plotly.graph_objects as go
import streamlit as st

import streamlit.components.v1 as components

from parallax import hardware, live_panel, telemetry_server
from parallax.spacecraft import (
    FAULT_DEFINITIONS, SpacecraftState, add_noise, clear_fault, copy_state, fault_details,
    inject_fault,
)
from parallax.fdir import run_fdir, FDIRReport
from parallax.gemma import run_predictive_analysis, live_status as gemma_live_status
from parallax.config import MODEL_NAME, USE_LIVE_GEMMA
from parallax.models import GemmaPredictiveAnalysis
from parallax.satellite_view import build_satellite_figure

st.set_page_config(page_title="PARALLAX FDIR", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = "#f4f6fb"
WHITE    = "#ffffff"
BORDER   = "#e2e8f0"
BORDER_S = "#cbd5e1"
TEXT     = "#1e293b"
TEXT_M   = "#64748b"
TEXT_D   = "#94a3b8"
NAVY     = "#1e3a6e"
BLUE     = "#2563eb"
GREEN    = "#059669"
RED      = "#dc2626"
AMBER    = "#d97706"
PURPLE   = "#7c3aed"

HEALTH_COLOR = {"nominal": GREEN, "degraded": AMBER, "failed": RED, "recovering": BLUE}
HEALTH_ICON  = {"nominal": "●", "degraded": "◑", "failed": "○", "recovering": "↻"}
FAULT_COLOR  = {
    "solar_string_loss":    AMBER,
    "pcu_fault":            "#ea580c",
    "reaction_wheel_fault": PURPLE,
    "spectrometer_fault":   BLUE,
    "comms_dropout":        RED,
    "thermal_runaway":      "#dc2626",
}
PHASE_COLOR  = {"detection": RED, "isolation": AMBER, "recovery": BLUE, "monitoring": GREEN}
# Fixed axis order for hardware series. Red and green stay reserved for health
# status, so the X/Y/Z triple uses blue/amber/purple (validated for CVD).
AXIS_COLOR   = {"X": BLUE, "Y": AMBER, "Z": PURPLE}
OUTCOME_SYM  = {"success": "✓", "partial": "◑", "in_progress": "…", "failed": "✗"}

st.markdown(f"""<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
.main .block-container {{ padding: 1.4rem 2.8rem 1.4rem; max-width: 100%; }}
#MainMenu, footer, header {{ display: none; }}

/* Fault trigger buttons */
.stButton button {{
    background: {WHITE} !important;
    border: 1.5px solid {BORDER_S} !important;
    color: {TEXT_M} !important;
    font-size: 0.78em !important;
    font-weight: 600 !important;
    padding: 12px 8px !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    transition: all 0.15s !important;
    line-height: 1.3 !important;
}}
.stButton button:hover {{
    border-color: {BLUE} !important;
    color: {BLUE} !important;
    box-shadow: 0 3px 8px rgba(37,99,235,0.15) !important;
    transform: translateY(-1px) !important;
}}
/* An active fault — click again to clear just that one */
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {{
    background: {RED}0f !important;
    border-color: {RED} !important;
    color: {RED} !important;
}}
.stButton button[kind="primary"]:hover,
.stButton button[data-testid="baseButton-primary"]:hover {{
    background: {RED}1f !important;
    border-color: {RED} !important;
    color: {RED} !important;
    box-shadow: 0 3px 8px rgba(220,38,38,0.18) !important;
}}

/* Expander */
details summary {{
    background: {WHITE};
    border: 1.5px solid {BORDER_S};
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 700;
    font-size: 0.85em;
    color: {TEXT};
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
details[open] summary {{ border-radius: 10px 10px 0 0; border-bottom: none; }}
details > div {{
    background: {WHITE};
    border: 1.5px solid {BORDER_S};
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 16px;
}}

div[data-testid="stVerticalBlock"] > div {{ gap: 0.35rem; }}
hr {{ border: none; border-top: 1.5px solid {BORDER}; margin: 8px 0; }}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _card(body, left_color=None, bg=WHITE, padding="11px 14px"):
    ls = f"border-left: 4px solid {left_color};" if left_color else ""
    return (f'<div style="background:{bg};border:1px solid {BORDER};border-radius:8px;'
            f'padding:{padding};margin:4px 0;box-shadow:0 1px 4px rgba(0,0,0,0.05);{ls}">'
            f'{body}</div>')


def _badge(text, color, sm=False):
    sz = "0.7em" if sm else "0.76em"
    return (f'<span style="background:{color}15;border:1px solid {color}40;color:{color};'
            f'padding:2px 9px;border-radius:20px;font-size:{sz};font-weight:700;">{text}</span>')


def _bar(pct, color, width=96):
    w = max(0, min(width, int(pct * width / 100)))
    return (
        f'<span style="display:inline-block;vertical-align:middle;'
        f'width:{width}px;height:7px;background:{BORDER};border-radius:4px;overflow:hidden;">'
        f'<span style="display:block;width:{w}px;height:7px;background:{color};border-radius:4px;"></span></span>'
        f'&nbsp;<span style="font-size:0.8em;font-weight:700;color:{color};">{pct}%</span>'
    )


def _section_label(text):
    st.markdown(
        f'<div style="font-size:0.68em;font-weight:700;color:{TEXT_D};letter-spacing:0.12em;'
        f'margin-bottom:10px;">{text}</div>',
        unsafe_allow_html=True,
    )


# ── State helpers ─────────────────────────────────────────────────────────────

def _toggle_fault(fault_id: str):
    """Break another part of the spacecraft, or repair just this one.

    Faults accumulate: several can be active at once and their effects compound
    in the shared state.
    """
    state = copy_state(st.session_state.spacecraft)
    if fault_id in state.active_faults:
        new_state = clear_fault(state, fault_id)
    else:
        new_state = inject_fault(state, fault_id)

    st.session_state.spacecraft    = new_state
    st.session_state.active_faults = list(new_state.active_faults)
    st.session_state.prediction    = None

    if new_state.active_faults:
        st.session_state.fdir_report  = run_fdir(new_state)   # instant, no AI
        st.session_state.gemma_needed = True
    else:
        st.session_state.fdir_report  = None
        st.session_state.gemma_needed = False


def _reset():
    st.session_state.spacecraft    = SpacecraftState()
    st.session_state.fdir_report   = None
    st.session_state.prediction    = None
    st.session_state.active_faults = []
    st.session_state.gemma_needed  = False


def _fault_buttons(active_faults, key_prefix):
    """Six always-available injectors; active ones are highlighted toggles."""
    fcols = st.columns(6)
    for i, (fid, fdef) in enumerate(FAULT_DEFINITIONS.items()):
        is_active = fid in active_faults
        with fcols[i]:
            label = f'{"✓" if is_active else fdef["icon"]}\n{fdef["label"]}'
            if st.button(
                label,
                key=f"{key_prefix}_{fid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help="Click to repair this fault" if is_active else "Click to inject this fault",
            ):
                _toggle_fault(fid)
                st.rerun()


def _call_gemma(state, fdir):
    fids = list(st.session_state.active_faults)
    state_dict = {k: v for k, v in state.__dict__.items() if not k.startswith("_")}
    fdir_dict = {
        "triggered": fdir.triggered,
        "active_faults": fdir.active_faults,
        "actions": [{"timestamp": a.timestamp, "phase": a.phase,
                     "message": a.message, "outcome": a.outcome} for a in fdir.actions],
        "mission_safe": fdir.mission_safe,
        "summary": fdir.summary,
    }
    try:
        pred, is_live = run_predictive_analysis(
            state_dict, fdir_dict,
            "Asteria-7, Jupiter approach cruise, Earth delay 38 minutes, autonomous operations required.",
            fids,
        )
        st.session_state.prediction = pred
        st.session_state.gemma_live = is_live
        st.session_state.gemma_error = None if is_live else gemma_live_status()
    except Exception as exc:
        st.session_state.gemma_error = str(exc)
        st.session_state.gemma_live = False
    st.session_state.gemma_needed = False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "spacecraft" not in st.session_state:
        st.session_state.spacecraft    = SpacecraftState()
        st.session_state.fdir_report   = None
        st.session_state.prediction    = None
        st.session_state.active_faults = []
        st.session_state.gemma_needed  = False
        st.session_state.tick          = 0

    st.session_state.tick += 1
    state: SpacecraftState       = st.session_state.spacecraft
    noisy                        = add_noise(copy_state(state), seed=st.session_state.tick)
    fdir: FDIRReport             = st.session_state.fdir_report
    pred: GemmaPredictiveAnalysis = st.session_state.prediction
    active_faults                = list(state.active_faults)
    fault_active                 = bool(active_faults)
    gemma_needed                 = st.session_state.gemma_needed

    # ── HEADER ───────────────────────────────────────────────────────────────
    ha, hb, hc = st.columns([1, 2, 1])
    with ha:
        st.markdown(
            f'<div style="font-weight:800;color:{NAVY};font-size:1.65em;letter-spacing:0.04em;line-height:1.1;">PARALLAX</div>'
            f'<div style="color:{TEXT_D};font-size:0.68em;font-weight:600;letter-spacing:0.12em;margin-top:1px;">FDIR + GEMMA INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )
    with hb:
        st.markdown(
            f'<div style="text-align:center;padding-top:12px;color:{TEXT_D};font-size:0.74em;'
            f'font-weight:500;letter-spacing:0.05em;">'
            f'ASTERIA-7 &nbsp;·&nbsp; JUPITER APPROACH &nbsp;·&nbsp; EARTH DELAY 38 MIN &nbsp;·&nbsp; AUTONOMOUS OPS</div>',
            unsafe_allow_html=True,
        )
    with hc:
        if len(active_faults) > 1:
            badge = _badge(f"⚡ {len(active_faults)} CONCURRENT FAULTS", RED)
        elif fault_active:
            badge = _badge("⚡ FAULT DETECTED", RED)
        else:
            badge = _badge("● ALL SYSTEMS NOMINAL", GREEN)
        st.markdown(f'<div style="text-align:right;padding-top:12px;">{badge}</div>', unsafe_allow_html=True)

    st.markdown("---")

    tab_mission, tab_hardware = st.tabs(["◈  MISSION CONTROL", "⚙  HARDWARE LINK"])
    with tab_mission:
        _render_mission_control(state, noisy, fdir, pred, active_faults, gemma_needed)
    with tab_hardware:
        _render_hardware_link()

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="color:{TEXT_D};font-size:0.7em;text-align:center;">'
        f'PARALLAX · Autonomous FDIR with Gemma predictive intelligence · Asteria-7 deep-space mission</div>',
        unsafe_allow_html=True,
    )


def _render_mission_control(state, noisy, fdir, pred, active_faults, gemma_needed):
    fault_active = bool(active_faults)

    # ── LIVE SENSOR STRIP ─────────────────────────────────────────────────────
    sensors = [
        ("Solar",    f"{noisy.solar_output_w:.0f} W",          noisy.solar_output_w < 650),
        ("Battery",  f"{noisy.battery_soc_pct:.1f}%",          noisy.battery_soc_pct < 80),
        ("Bus V",    f"{noisy.bus_voltage_v:.2f} V",           noisy.bus_voltage_v < 27.0),
        ("PCU Temp", f"{noisy.pcu_temp_c:.1f}°C",              noisy.pcu_temp_c > 50),
        ("Signal",   f"{noisy.signal_strength_dbm:.1f} dBm",   noisy.signal_strength_dbm < -121),
        ("RW Speed", f"{noisy.rw_speed_rpm:.0f} rpm",          noisy.rw_speed_rpm < 600 or noisy.rw_speed_rpm > 4500),
        ("Attitude", f"{noisy.attitude_error_arcsec:.2f}\"",   noisy.attitude_error_arcsec > 5),
        ("Data Rate",f"{noisy.data_rate_mbps:.3f} Mbps",       noisy.data_rate_mbps < 0.2),
    ]
    scols = st.columns(8)
    for i, (label, value, is_alert) in enumerate(sensors):
        vc = RED if is_alert else TEXT
        with scols[i]:
            st.markdown(
                f'<div style="background:{WHITE};border:1px solid {BORDER};border-radius:8px;'
                f'padding:8px 10px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
                f'<div style="font-size:0.62em;font-weight:700;color:{TEXT_D};letter-spacing:0.08em;margin-bottom:3px;">{label.upper()}</div>'
                f'<div style="font-size:0.9em;font-weight:700;color:{vc};font-family:monospace;">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── NOMINAL STATE ─────────────────────────────────────────────────────────
    if not fault_active:
        st.markdown(
            f'<div style="color:{TEXT_D};font-size:0.7em;font-weight:700;letter-spacing:0.12em;margin-bottom:8px;">'
            f'INJECT FAULT EVENTS — SELECT ONE OR MORE SUBSYSTEM FAILURES:</div>',
            unsafe_allow_html=True,
        )
        _fault_buttons(active_faults, "btn")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _, mcol, _ = st.columns([1, 2.2, 1])
        with mcol:
            st.plotly_chart(
                build_satellite_figure(noisy),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f'<div style="text-align:center;color:{TEXT_D};font-size:0.78em;margin-top:-6px;">'
                f'Select one or more faults above to see FDIR recovery and Gemma AI analysis in real time</div>',
                unsafe_allow_html=True,
            )

    # ── FAULT ACTIVE STATE ────────────────────────────────────────────────────
    else:
        st.markdown(
            f'<div style="color:{TEXT_D};font-size:0.7em;font-weight:700;letter-spacing:0.12em;margin-bottom:8px;">'
            f'ACTIVE FAULTS — CLICK TO BREAK ANOTHER PART, OR CLICK A HIGHLIGHTED FAULT TO REPAIR IT:</div>',
            unsafe_allow_html=True,
        )
        _fault_buttons(active_faults, "btn")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        rc, _ = st.columns([1, 8])
        with rc:
            if st.button("↩ Reset all", type="secondary"):
                _reset()
                st.rerun()

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        left_col, mid_col, right_col = st.columns([1.1, 1.55, 1.35])

        # ── LEFT: FDIR (runs instantly, zero AI) ─────────────────────────────
        with left_col:
            _section_label("FDIR AUTONOMOUS RECOVERY")

            # One identity card per concurrently active fault
            for fid in active_faults:
                fd = fault_details(state, fid)
                fc = FAULT_COLOR.get(fid, AMBER)
                st.markdown(_card(
                    f'<div style="color:{fc};font-weight:700;font-size:0.83em;margin-bottom:4px;">⚡ {fd.get("label","").upper()}</div>'
                    f'<div style="color:{TEXT_M};font-size:0.75em;line-height:1.45;">{fd.get("description","")}</div>',
                    left_color=fc,
                ), unsafe_allow_html=True)

            # FDIR timeline — colour-coded by the fault each action belongs to
            if fdir and fdir.triggered:
                for a in fdir.actions:
                    c   = PHASE_COLOR.get(a.phase, TEXT_D)
                    sym = OUTCOME_SYM.get(a.outcome, "")
                    edge = FAULT_COLOR.get(a.fault_id, c) if len(active_faults) > 1 else c
                    msg = a.message
                    for prefix in ("RECOVERY: ", "ISOLATION: "):
                        if msg.startswith(prefix):
                            msg = msg[len(prefix):]
                    st.markdown(
                        f'<div style="border-left:3px solid {edge}55;padding:4px 10px;margin:2px 0;">'
                        f'<span style="color:{TEXT_D};font-size:0.67em;font-family:monospace;">[{a.timestamp}]</span>&nbsp;'
                        f'<span style="color:{c};font-size:0.73em;font-weight:600;">{sym} {msg}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                safe_c = GREEN if fdir.mission_safe else RED
                safe_l = ("MISSION SAFE — CONTINGENCY MODE"
                          if fdir.mission_safe else "⚠ MISSION CRITICAL")
                st.markdown(
                    f'<div style="margin-top:12px;padding:9px 14px;background:{safe_c}10;'
                    f'border:1.5px solid {safe_c}40;border-radius:8px;font-size:0.76em;'
                    f'font-weight:700;color:{safe_c};text-align:center;">{safe_l}</div>',
                    unsafe_allow_html=True,
                )

        # ── CENTRE: 3D MODEL + HEALTH ─────────────────────────────────────────
        with mid_col:
            _section_label("SPACECRAFT STATE")

            st.plotly_chart(
                build_satellite_figure(noisy),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Subsystem health grid
            hcols = st.columns(3)
            for i, (sname, hstatus) in enumerate(noisy.subsystem_health.items()):
                hc = HEALTH_COLOR.get(hstatus, TEXT_D)
                hi = HEALTH_ICON.get(hstatus, "?")
                with hcols[i % 3]:
                    st.markdown(
                        f'<div style="background:{hc}12;border:1px solid {hc}30;border-radius:6px;'
                        f'padding:5px 8px;text-align:center;margin:2px 0;">'
                        f'<span style="color:{hc};font-size:0.7em;font-weight:700;">{hi} {sname}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── RIGHT: GEMMA (spinner while loading, results when ready) ──────────
        with right_col:
            _section_label("GEMMA INTELLIGENCE LAYER")

            # Say plainly where this analysis came from. A cached answer shown
            # as a live one misrepresents what the system actually did.
            if pred is not None:
                if st.session_state.get("gemma_live"):
                    st.markdown(
                        f'<div style="margin-bottom:6px;">{_badge(f"◆ LIVE — {MODEL_NAME}", GREEN, sm=True)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    reason = st.session_state.get("gemma_error")
                    label = "CACHED — LIVE CALLS OFF" if not USE_LIVE_GEMMA else "CACHED — LIVE CALL FAILED"
                    st.markdown(
                        f'<div style="margin-bottom:6px;">{_badge("⛃ " + label, AMBER, sm=True)}</div>'
                        + (f'<div style="font-size:0.68em;color:{TEXT_D};line-height:1.45;'
                           f'margin:-2px 0 6px;">{reason}</div>' if reason else ""),
                        unsafe_allow_html=True,
                    )

            if gemma_needed and pred is None:
                with st.spinner("Gemma analysing the situation..."):
                    _call_gemma(state, fdir)
                st.rerun()

            elif pred is None and not gemma_needed:
                st.markdown(_card(
                    f'<div style="color:{TEXT_D};font-size:0.8em;text-align:center;padding:20px 0;">Awaiting analysis</div>',
                ), unsafe_allow_html=True)

            else:
                # Stability banner
                stab_c = {"stable": GREEN, "degraded": AMBER,
                          "critical": RED, "unknown": TEXT_D}.get(pred.system_stability, TEXT_D)
                earth_d    = getattr(pred, "earth_delay_min",       38)
                basic_pct  = getattr(pred, "basic_fix_success_pct", 70)
                cascade_pct = getattr(pred, "cascade_failure_pct",   25)
                basic_c    = GREEN if basic_pct  >= 75 else (AMBER if basic_pct  >= 50 else RED)
                casc_c     = RED   if cascade_pct >= 50 else (AMBER if cascade_pct >= 25 else GREEN)

                st.markdown(_card(
                    f'<div style="color:{stab_c};font-weight:700;font-size:0.82em;margin-bottom:10px;">'
                    f'SYSTEM STABILITY: {pred.system_stability.upper()}</div>'
                    f'<div style="font-size:0.74em;color:{TEXT_M};margin:5px 0;">'
                    f'Earth signal delay&nbsp;&nbsp;<strong style="color:{TEXT};">{earth_d} min one-way</strong></div>'
                    f'<div style="font-size:0.74em;color:{TEXT_M};margin:5px 0;">'
                    f'Basic FDIR success&nbsp;&nbsp;&nbsp;{_bar(basic_pct,  basic_c)}</div>'
                    f'<div style="font-size:0.74em;color:{TEXT_M};margin:5px 0;">'
                    f'Cascade failure risk {_bar(cascade_pct, casc_c)}</div>',
                    left_color=stab_c,
                ), unsafe_allow_html=True)

                # Short assessment
                blurb = pred.current_assessment
                if len(blurb) > 230:
                    blurb = blurb[:227] + "…"
                st.markdown(
                    f'<div style="font-size:0.74em;color:{TEXT_M};line-height:1.55;'
                    f'border-left:3px solid {BORDER_S};padding:6px 10px;margin:6px 0;">{blurb}</div>',
                    unsafe_allow_html=True,
                )

                # Fix options
                alt_fixes = getattr(pred, "alternative_fixes", [])
                chosen    = getattr(pred, "chosen_fix", "")
                if alt_fixes:
                    st.markdown(
                        f'<div style="font-size:0.68em;font-weight:700;color:{TEXT_D};'
                        f'letter-spacing:0.1em;margin:8px 0 4px;">FIX OPTIONS</div>',
                        unsafe_allow_html=True,
                    )
                    for fix in alt_fixes:
                        is_chosen = fix.name == chosen
                        fpct  = getattr(fix, "success_pct", 0)
                        fc2   = GREEN if fpct >= 75 else (AMBER if fpct >= 50 else RED)
                        bc    = BLUE if is_chosen else BORDER
                        bg    = "#eff6ff" if is_chosen else WHITE
                        sel   = f'&nbsp;{_badge("SELECTED", BLUE, sm=True)}' if is_chosen else ""
                        auto  = _badge("AUTO",   GREEN, sm=True) if fix.autonomous else _badge("MANUAL", AMBER, sm=True)
                        st.markdown(
                            f'<div style="background:{bg};border:1.5px solid {bc};border-radius:8px;'
                            f'padding:8px 12px;margin:4px 0;">'
                            f'<div style="font-size:0.78em;font-weight:700;color:{TEXT};margin-bottom:3px;">'
                            f'{fix.name}{sel}</div>'
                            f'<div style="font-size:0.7em;color:{TEXT_M};margin-bottom:6px;line-height:1.4;">'
                            f'{fix.description}</div>'
                            f'<div>{_bar(fpct, fc2, 80)}&nbsp;'
                            f'<span style="font-size:0.66em;color:{TEXT_D};">[{fix.risk.upper()} RISK]</span>'
                            f'&nbsp;{auto}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Decision
                execute      = getattr(pred, "execute_immediately", True)
                reason       = getattr(pred, "execute_reason", "")
                chosen_name  = getattr(pred, "chosen_fix", "Basic FDIR Recovery")
                dc  = GREEN if execute else AMBER
                dl  = f"EXECUTE: {chosen_name}" if execute else f"HOLD — MONITOR: {chosen_name}"
                st.markdown(
                    f'<div style="background:{dc}10;border:2px solid {dc}40;border-radius:8px;'
                    f'padding:10px 14px;margin-top:10px;">'
                    f'<div style="font-size:0.68em;font-weight:700;color:{TEXT_D};margin-bottom:4px;'
                    f'letter-spacing:0.08em;">GEMMA DECISION</div>'
                    f'<div style="font-weight:700;color:{dc};font-size:0.82em;">{dl}</div>'
                    + (f'<div style="font-size:0.72em;color:{TEXT_M};margin-top:5px;">{reason}</div>'
                       if reason else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )

        # ── DEEP DIVE ─────────────────────────────────────────────────────────
        if pred is not None:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            with st.expander("📊 Deep Dive — Predicted Failures, Cascading Risks & Earth Transmission Report"):
                d1, d2, d3 = st.columns([1, 1, 1])

                with d1:
                    st.markdown(f'<div style="font-weight:700;color:{TEXT};font-size:0.88em;margin-bottom:10px;">Predicted Next Failures</div>', unsafe_allow_html=True)
                    if pred.predicted_failures:
                        for pf in pred.predicted_failures:
                            pc = RED if pf.probability == "high" else (AMBER if pf.probability == "medium" else GREEN)
                            warnings = "".join(f'<div style="margin-top:2px;">· {w}</div>' for w in pf.early_warning_signs)
                            st.markdown(
                                f'<div style="background:{WHITE};border:1px solid {BORDER};'
                                f'border-left:4px solid {pc};border-radius:6px;padding:10px 12px;margin:6px 0;">'
                                f'<div style="font-weight:700;font-size:0.82em;color:{TEXT};margin-bottom:3px;">{pf.subsystem}</div>'
                                f'<div style="font-size:0.75em;color:{TEXT_M};margin-bottom:5px;">{pf.failure_mode}</div>'
                                f'<div style="font-size:0.72em;margin-bottom:5px;">'
                                f'{_badge(pf.probability.upper()+" RISK", pc, sm=True)}'
                                f'&nbsp;&nbsp;<span style="color:{TEXT_D};">⏱ {pf.estimated_time_to_failure}</span></div>'
                                f'<div style="font-size:0.7em;color:{TEXT_D};line-height:1.5;">{warnings}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(f'<div style="color:{TEXT_D};font-size:0.8em;">No failures predicted at this time.</div>', unsafe_allow_html=True)

                with d2:
                    st.markdown(f'<div style="font-weight:700;color:{TEXT};font-size:0.88em;margin-bottom:10px;">Cascading Risks</div>', unsafe_allow_html=True)
                    for risk in pred.cascading_risks:
                        st.markdown(
                            f'<div style="background:{AMBER}0a;border:1px solid {AMBER}30;'
                            f'border-left:4px solid {AMBER};border-radius:6px;padding:8px 12px;'
                            f'margin:5px 0;font-size:0.77em;color:{TEXT_M};line-height:1.45;">⚠ {risk}</div>',
                            unsafe_allow_html=True,
                        )
                    if not pred.cascading_risks:
                        st.markdown(f'<div style="color:{TEXT_D};font-size:0.8em;">No cascading risks identified.</div>', unsafe_allow_html=True)

                    st.markdown(f'<div style="font-weight:700;color:{TEXT};font-size:0.88em;margin-top:18px;margin-bottom:10px;">Recommended Actions</div>', unsafe_allow_html=True)
                    for i, action in enumerate(pred.recommended_actions, 1):
                        st.markdown(
                            f'<div style="background:{BLUE}08;border:1px solid {BLUE}25;'
                            f'border-left:4px solid {BLUE};border-radius:6px;padding:8px 12px;'
                            f'margin:4px 0;font-size:0.77em;color:{TEXT_M};line-height:1.45;">'
                            f'<strong style="color:{BLUE};">{i}.</strong> {action}</div>',
                            unsafe_allow_html=True,
                        )

                with d3:
                    st.markdown(f'<div style="font-weight:700;color:{TEXT};font-size:0.88em;margin-bottom:10px;">Earth Transmission Report</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#f8fafc;border:1px solid {BORDER};border-radius:8px;'
                        f'padding:14px 16px;font-family:monospace;font-size:0.72em;color:{TEXT_M};'
                        f'white-space:pre-wrap;line-height:1.65;max-height:340px;overflow-y:auto;">'
                        f'{pred.earth_report}</div>',
                        unsafe_allow_html=True,
                    )
                    conf_c = GREEN if pred.confidence == "high" else (AMBER if pred.confidence == "medium" else RED)
                    st.markdown(
                        f'<div style="margin-top:8px;font-size:0.74em;color:{TEXT_D};">'
                        f'Confidence: {_badge(pred.confidence.upper(), conf_c, sm=True)}</div>',
                        unsafe_allow_html=True,
                    )

# ── Hardware link tab ─────────────────────────────────────────────────────────

@st.cache_resource
def _serial_link() -> hardware.SerialLink:
    """One reader thread for the whole server, surviving Streamlit reruns."""
    return hardware.SerialLink()


@st.cache_resource
def _feed_port() -> int:
    """Loopback port the browser panel polls. Started once per server."""
    return telemetry_server.start_server(_serial_link())


FACE_COLORS = {
    "+X": BLUE,   "-X": "#bcd0fa",
    "+Y": AMBER,  "-Y": "#f8dfb8",
    "+Z": PURPLE, "-Z": "#d9c6fb",
}


CHART_MAX_POINTS = 400


def _timeseries(samples, series, y_title, zero_line=False):
    """Line chart over the sample buffer. `series` is [(label, attribute, color)]."""
    # At 50 Hz the buffer holds a few thousand frames; drawing them all would
    # cost more than it shows. Stride down to a sane number of points.
    stride = max(1, len(samples) // CHART_MAX_POINTS)
    samples = samples[::stride]

    now = time.time()
    age = [s.received_at - now for s in samples]

    fig = go.Figure()
    for label, attribute, color in series:
        values = [getattr(s, attribute) for s in samples]
        fig.add_trace(go.Scatter(
            x=age, y=values, mode="lines", name=label,
            line={"color": color, "width": 2},
            hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
        ))

    if zero_line:
        fig.add_hline(y=0, line={"color": BORDER_S, "width": 1})

    # Margins leave room for tick labels and keep the x title clear of them;
    # units live in the section heading, so the y axis carries ticks only.
    fig.update_layout(
        height=210, margin={"l": 46, "r": 12, "t": 30, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=len(series) > 1,
        legend={"orientation": "h", "y": 1.22, "x": 0, "font": {"size": 10, "color": TEXT_M},
                "bgcolor": "rgba(0,0,0,0)"},
        hovermode="x unified",
        xaxis={"title": {"text": "seconds ago", "font": {"size": 9, "color": TEXT_D}, "standoff": 6},
               "gridcolor": BORDER, "zeroline": False, "tickfont": {"size": 9, "color": TEXT_D},
               "linecolor": BORDER_S, "ticks": "outside", "ticklen": 3, "tickcolor": BORDER_S},
        yaxis={"title": {"text": y_title, "font": {"size": 9, "color": TEXT_D}, "standoff": 4},
               "gridcolor": BORDER, "zeroline": False, "tickfont": {"size": 9, "color": TEXT_D},
               "linecolor": BORDER_S, "nticks": 5},
        font={"family": "monospace"},
    )
    return fig


def _render_hardware_link():
    link = _serial_link()

    if not hardware.SERIAL_AVAILABLE:
        st.markdown(_card(
            f'<div style="color:{RED};font-weight:700;font-size:0.82em;margin-bottom:4px;">PYSERIAL NOT INSTALLED</div>'
            f'<div style="color:{TEXT_M};font-size:0.75em;">Run <code>pip install pyserial</code> and restart the dashboard.</div>',
            left_color=RED,
        ), unsafe_allow_html=True)
        return

    # ── Connection controls ──────────────────────────────────────────────────
    _section_label("BOARD CONNECTION")
    ports = hardware.list_ports()
    port_ids = [device for device, _ in ports]

    c1, cb, c2, c3, c4 = st.columns([2, 1, 1, 1, 2.4])
    with c1:
        if port_ids:
            default = port_ids.index(link.port) if link.port in port_ids else 0
            port = st.selectbox("Serial port", port_ids, index=default,
                                format_func=lambda d: f"{d}  ·  {dict(ports)[d]}",
                                label_visibility="collapsed")
        else:
            port = None
            st.markdown(f'<div style="color:{TEXT_D};font-size:0.78em;padding-top:6px;">No serial ports found</div>',
                        unsafe_allow_html=True)
    with cb:
        baud = st.selectbox("Baud", hardware.BAUD_OPTIONS, index=0,
                            format_func=lambda b: f"{b} baud", label_visibility="collapsed",
                            help="115200 for sketch_aug1c. Use 460800 after flashing "
                                 "parallax_handle, which streams at 50 Hz.")
    with c2:
        if st.button("▶ Connect", key="hw_connect", use_container_width=True, disabled=port is None):
            link.start(port, baud)
            time.sleep(1.2)   # let the reader thread bind the port or fail
            st.rerun()
    with c3:
        if st.button("■ Disconnect", key="hw_disconnect", use_container_width=True,
                     disabled=not link.is_running()):
            link.stop()
            st.rerun()
    with c4:
        if link.is_live():
            status = _badge(f"● STREAMING · {link.sample_rate_hz():.1f} Hz", GREEN)
        elif link.is_running():
            status = _badge("◑ CONNECTED — WAITING FOR DATA", AMBER)
        else:
            status = _badge("○ DISCONNECTED", TEXT_D)
        st.markdown(f'<div style="padding-top:6px;">{status}</div>', unsafe_allow_html=True)

    if link.error:
        st.markdown(_card(
            f'<div style="color:{RED};font-weight:700;font-size:0.8em;margin-bottom:3px;">SERIAL LINK ERROR</div>'
            f'<div style="color:{TEXT_M};font-size:0.75em;line-height:1.45;">{link.error}</div>',
            left_color=RED,
        ), unsafe_allow_html=True)
    elif link.is_running() and link.board_message and not link.is_live():
        st.markdown(_card(
            f'<div style="color:{TEXT_M};font-size:0.75em;">Board says: <code>{link.board_message}</code></div>',
        ), unsafe_allow_html=True)

    if not link.is_running():
        st.markdown(
            f'<div style="color:{TEXT_D};font-size:0.78em;margin-top:10px;line-height:1.6;">'
            f'Select the board\'s port and press <strong>Connect</strong>. The Arduino IDE Serial Monitor '
            f'must be closed — only one program can hold the port at a time.<br>'
            f'Expected firmware: <code>sketch_aug1c</code> (MPU6050 gyro + accel, DS18B20 temperature, HC-SR04 range) '
            f'at {hardware.BAUD_RATE} baud.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _section_label("BOARD ORIENTATION — LIVE ATTITUDE")

    # The live panel is a component, not a fragment: Streamlit repaints by
    # rerunning on the server, which remounts the chart and reads as a flicker.
    # This iframe mounts once and animates itself from the loopback feed, so
    # nothing on the Python side has to rerun to keep it moving.
    components.html(
        live_panel.build_panel_html(
            _feed_port(),
            {"BG": BG, "WHITE": WHITE, "BORDER": BORDER, "BORDER_S": BORDER_S,
             "TEXT": TEXT, "TEXT_M": TEXT_M, "TEXT_D": TEXT_D,
             "GREEN": GREEN, "AMBER": AMBER, "RED": RED},
            FACE_COLORS, AXIS_COLOR,
        ),
        height=live_panel.PANEL_HEIGHT,
    )

    _render_traces()


@st.fragment(run_every=0.8)
def _render_traces():
    """Slow fragment — history charts, spacecraft sync and the raw frame table."""
    link = _serial_link()
    latest = link.latest()
    samples = link.history()
    if latest is None:
        return

    # ── Traces ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    gcol, acol = st.columns(2)
    with gcol:
        _section_label("ANGULAR RATE — MPU6050 GYRO (°/s)")
        st.plotly_chart(
            _timeseries(samples, [("Gyro X", "gyro_x", AXIS_COLOR["X"]),
                                  ("Gyro Y", "gyro_y", AXIS_COLOR["Y"]),
                                  ("Gyro Z", "gyro_z", AXIS_COLOR["Z"])],
                        "°/s", zero_line=True),
            use_container_width=True, config={"displayModeBar": False},
        )
    with acol:
        _section_label("SPECIFIC FORCE — MPU6050 ACCELEROMETER (g)")
        st.plotly_chart(
            _timeseries(samples, [("Accel X", "accel_x", AXIS_COLOR["X"]),
                                  ("Accel Y", "accel_y", AXIS_COLOR["Y"]),
                                  ("Accel Z", "accel_z", AXIS_COLOR["Z"])],
                        "g", zero_line=True),
            use_container_width=True, config={"displayModeBar": False},
        )

    ranged = [s for s in samples if s.distance_cm is not None]
    if len(ranged) > 1:
        _section_label("ULTRASONIC RANGE — HC-SR04 (cm)")
        st.plotly_chart(
            _timeseries(ranged, [("Range", "distance_cm", NAVY)], "cm"),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Spacecraft sync ──────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    mapped = hardware.map_to_spacecraft(latest)
    synced = st.toggle(
        "Drive spacecraft ADCS telemetry from this board",
        key="hw_sync",
        help="Body rate becomes attitude error, board tilt becomes Sun-pointing error, "
             "and the DS18B20 (when present) becomes instrument temperature.",
    )
    if synced:
        hardware.apply_to_state(st.session_state.spacecraft, latest)

    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;font-size:0.74em;'
        f'color:{TEXT_M};padding:3px 0;">'
        f'<span>{name.replace("_", " ")}</span>'
        f'<strong style="color:{TEXT};font-family:monospace;">{value:.3f}</strong></div>'
        for name, value in mapped.items()
    )
    st.markdown(_card(
        f'<div style="font-size:0.68em;font-weight:700;color:{TEXT_D};letter-spacing:0.1em;'
        f'margin-bottom:6px;">MAPPED SPACECRAFT TELEMETRY '
        f'{"— LIVE ON MISSION CONTROL" if synced else "— PREVIEW ONLY"}</div>{rows}',
        left_color=GREEN if synced else BORDER_S,
    ), unsafe_allow_html=True)

    with st.expander("Raw frames"):
        st.dataframe(
            [{"age_s": round(s.received_at - time.time(), 2),
              "gyro_x": s.gyro_x, "gyro_y": s.gyro_y, "gyro_z": s.gyro_z,
              "accel_x": s.accel_x, "accel_y": s.accel_y, "accel_z": s.accel_z,
              "temp_c": s.temp_c, "distance_cm": s.distance_cm}
             for s in reversed(samples[-40:])],
            use_container_width=True, hide_index=True,
        )


if __name__ == "__main__":
    main()
