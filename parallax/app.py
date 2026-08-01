import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from parallax import hardware, live_panel, telemetry_server, stack_ui
from parallax.benchmark import SCENARIOS, run_all
from parallax.featurize import featurize_spacecraft_state
from parallax.fdir import run_fdir, FDIRReport
from parallax.gemma import live_status as gemma_live_status
from parallax.config import MODEL_NAME, USE_LIVE_GEMMA
from parallax.spacecraft import (
    FAULT_DEFINITIONS, SpacecraftState, add_noise, clear_fault, copy_state, fault_details,
    inject_fault,
)
from parallax.tiers import StackResult, run_sentinel, run_stack
from parallax.validator import apply_approved_plan

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
AXIS_COLOR   = {"X": BLUE, "Y": AMBER, "Z": PURPLE}
OUTCOME_SYM  = {"success": "✓", "partial": "◑", "in_progress": "…", "failed": "✗"}

st.markdown(f"""<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
.main .block-container {{ padding: 1.6rem 2.8rem 1.6rem; max-width: 100%; }}
#MainMenu, footer, header {{ display: none; }}

.stButton button {{
    background: {WHITE} !important;
    border: 1.5px solid {BORDER_S} !important;
    color: {TEXT_M} !important;
    font-size: 0.79em !important;
    font-weight: 600 !important;
    padding: 13px 8px !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
    transition: all 0.14s ease !important;
    line-height: 1.35 !important;
}}
.stButton button:hover {{
    border-color: {BLUE} !important;
    color: {BLUE} !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.16) !important;
    transform: translateY(-2px) !important;
}}
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {{
    background: {RED}0d !important;
    border-color: {RED}80 !important;
    color: {RED} !important;
    box-shadow: 0 2px 8px rgba(220,38,38,0.12) !important;
}}
.stButton button[kind="primary"]:hover,
.stButton button[data-testid="baseButton-primary"]:hover {{
    background: {RED}18 !important;
    border-color: {RED} !important;
    color: {RED} !important;
    box-shadow: 0 4px 12px rgba(220,38,38,0.2) !important;
    transform: translateY(-2px) !important;
}}

details summary {{
    background: {WHITE};
    border: 1.5px solid {BORDER_S};
    border-radius: 10px;
    padding: 13px 18px;
    font-weight: 700;
    font-size: 0.84em;
    color: {TEXT};
    cursor: pointer;
    box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    letter-spacing: 0.01em;
}}
details[open] summary {{ border-radius: 10px 10px 0 0; border-bottom: none; }}
details > div {{
    background: {WHITE};
    border: 1.5px solid {BORDER_S};
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 18px;
}}

div[data-testid="stVerticalBlock"] > div {{ gap: 0.5rem; }}
hr {{ border: none; border-top: 1.5px solid {BORDER}; margin: 10px 0; }}

iframe {{ display: block; }}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _card(body, left_color=None, bg=WHITE, padding="12px 14px"):
    ls = f"border-left: 4px solid {left_color};" if left_color else ""
    return (f'<div style="background:{bg};border:1px solid {BORDER};border-radius:10px;'
            f'padding:{padding};margin:4px 0;box-shadow:0 2px 6px rgba(0,0,0,0.06);{ls}">'
            f'{body}</div>')


def _badge(text, color, sm=False):
    sz = "0.7em" if sm else "0.76em"
    return (f'<span style="background:{color}15;border:1px solid {color}40;color:{color};'
            f'padding:2px 9px;border-radius:20px;font-size:{sz};font-weight:700;">{text}</span>')


def _section_label(text):
    st.markdown(
        f'<div style="font-size:0.67em;font-weight:800;color:{TEXT_D};letter-spacing:0.13em;'
        f'text-transform:uppercase;margin-bottom:10px;padding-bottom:6px;'
        f'border-bottom:1.5px solid {BORDER};">{text}</div>',
        unsafe_allow_html=True,
    )


# ── State helpers ─────────────────────────────────────────────────────────────

def _toggle_fault(fault_id: str):
    state = copy_state(st.session_state.spacecraft)
    if fault_id in state.active_faults:
        new_state = clear_fault(state, fault_id)
    else:
        new_state = inject_fault(state, fault_id)

    st.session_state.spacecraft    = new_state
    st.session_state.active_faults = list(new_state.active_faults)
    st.session_state.stack_result  = None
    st.session_state.pre_plan_state = None

    if new_state.active_faults:
        st.session_state.fdir_report  = run_fdir(new_state)
        st.session_state.stack_needed = True
    else:
        st.session_state.fdir_report  = None
        st.session_state.stack_needed = False


def _reset():
    st.session_state.spacecraft     = SpacecraftState()
    st.session_state.fdir_report    = None
    st.session_state.stack_result   = None
    st.session_state.active_faults  = []
    st.session_state.stack_needed   = False
    st.session_state.pre_plan_state = None
    st.session_state.plan_committed = False


def _fault_buttons(active_faults, key_prefix):
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


def _hardware_samples() -> list:
    link = _serial_link()
    return link.history() if link.is_running() else []


def _run_stack_now(state, fdir):
    """Fire the five-tier stack. G0 is skipped here — it runs on its own timer."""
    from parallax.benchmark import _synthetic_evidence

    samples = _hardware_samples()
    featurised = featurize_spacecraft_state(state, samples=samples)
    fids = list(st.session_state.active_faults)
    evidence = _synthetic_evidence(fids)

    try:
        result = run_stack(
            featurised_state=featurised,
            fdir_summary=fdir.summary if fdir else "",
            active_faults=fids,
            available_evidence=evidence,
            current_state=state,
            include_sentinel=False,
            max_replans=1,
        )
        st.session_state.stack_result = result
        st.session_state.pre_plan_state = copy_state(state)
        # If any tier reported an error the trace already carries it — we just
        # record a summary line for the status pill.
        errored = [t for t in result.traces if t.error]
        st.session_state.gemma_error = errored[0].error if errored else None
        st.session_state.gemma_live = all(t.is_live for t in result.traces if t.tier != "G3" or t.ok)
    except Exception as exc:
        st.session_state.stack_result = None
        st.session_state.gemma_error = f"{type(exc).__name__}: {exc}"
        st.session_state.gemma_live = False
    st.session_state.stack_needed = False


def _commit_plan():
    """Apply the currently approved plan to live spacecraft state and re-run FDIR."""
    result: StackResult | None = st.session_state.get("stack_result")
    if not result or not result.plan:
        return
    if not (result.validation and result.validation.approved and result.verdict and result.verdict.approved):
        return

    steps = [s.model_dump() for s in result.plan.steps]
    new_state = apply_approved_plan(steps, st.session_state.spacecraft)
    st.session_state.spacecraft = new_state
    st.session_state.active_faults = list(new_state.active_faults)
    st.session_state.fdir_report = run_fdir(new_state)
    st.session_state.plan_committed = True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()

    st.session_state.tick += 1
    state: SpacecraftState = st.session_state.spacecraft
    noisy                  = add_noise(copy_state(state), seed=st.session_state.tick)
    fdir: FDIRReport       = st.session_state.fdir_report
    stack_result: StackResult | None = st.session_state.stack_result
    active_faults          = list(state.active_faults)
    fault_active           = bool(active_faults)
    stack_needed           = st.session_state.stack_needed

    _render_header(active_faults, fault_active)
    st.markdown("---")
    _render_mission_control(state, noisy, fdir, stack_result, active_faults, stack_needed)

    # The capsule / benchmark deep-dives are advanced-only. In simplified mode
    # the stack panel above is the complete demo story.
    if _is_simplified():
        st.markdown(
            f'<div style="text-align:center;color:{TEXT_D};font-size:0.72em;margin-top:12px;">'
            f'Switch to <strong>Advanced</strong> to inspect deterministic-gate detail, the '
            f'G4 evidence capsule, and the FDIR-vs-stack benchmark harness.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("---")
        _render_bottom_row(state, stack_result)

    st.markdown("---")
    st.markdown(
        f'<div style="color:{TEXT_D};font-size:0.7em;text-align:center;">'
        f'PARALLAX · Five-role Gemma reasoning stack on Asteria-7 · G0 SENTINEL · G1 DIAGNOSTICIAN · '
        f'G2 FLIGHT DIRECTOR · G3 ADJUDICATOR · G4 ARCHIVIST</div>',
        unsafe_allow_html=True,
    )


def _init_state():
    defaults = {
        "spacecraft":     SpacecraftState(),
        "fdir_report":    None,
        "stack_result":   None,
        "active_faults":  [],
        "stack_needed":   False,
        "pre_plan_state": None,
        "plan_committed": False,
        "tick":           0,
        "sentinel_verdict":  None,
        "sentinel_ok":       False,
        "sentinel_last_run": 0.0,
        "capsule_budget_kb": 25,
        "view_mode":         "simplified",   # "simplified" | "advanced"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _is_simplified() -> bool:
    return st.session_state.get("view_mode", "simplified") == "simplified"


def _render_header(active_faults, fault_active):
    ha, hb, hc, hd = st.columns([1.1, 1.7, 0.9, 0.9])
    with ha:
        st.markdown(
            f'<div style="font-weight:900;color:{NAVY};font-size:1.8em;letter-spacing:0.05em;'
            f'line-height:1.05;">PARALLAX</div>'
            f'<div style="color:{TEXT_D};font-size:0.67em;font-weight:700;letter-spacing:0.14em;'
            f'margin-top:3px;">FIVE-ROLE GEMMA REASONING STACK</div>',
            unsafe_allow_html=True,
        )
    with hb:
        st.markdown(
            f'<div style="text-align:center;padding-top:14px;color:{TEXT_D};font-size:0.72em;'
            f'font-weight:600;letter-spacing:0.07em;">'
            f'ASTERIA-7 &nbsp;·&nbsp; JUPITER APPROACH &nbsp;·&nbsp; '
            f'EARTH DELAY 38 MIN &nbsp;·&nbsp; AUTONOMOUS OPS</div>',
            unsafe_allow_html=True,
        )
    with hc:
        # View-mode toggle. Radio is compact enough and reflects state clearly.
        mode = st.radio(
            "View",
            options=["Simplified", "Advanced"],
            index=0 if _is_simplified() else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="view_mode_radio",
        )
        new_mode = "simplified" if mode == "Simplified" else "advanced"
        if new_mode != st.session_state.get("view_mode"):
            st.session_state["view_mode"] = new_mode
            st.rerun()
    with hd:
        if len(active_faults) > 1:
            badge = _badge(f"⚡ {len(active_faults)} CONCURRENT FAULTS", RED)
        elif fault_active:
            badge = _badge("⚡ FAULT DETECTED", RED)
        else:
            badge = _badge("● ALL SYSTEMS NOMINAL", GREEN)
        st.markdown(
            f'<div style="text-align:right;padding-top:14px;">{badge}</div>',
            unsafe_allow_html=True,
        )


def _live_model(height: int = 580):
    palette = {"BG": WHITE, "WHITE": WHITE, "BORDER": BORDER, "BORDER_S": BORDER_S,
               "TEXT": TEXT, "TEXT_M": TEXT_M, "TEXT_D": TEXT_D,
               "GREEN": GREEN, "AMBER": AMBER, "RED": RED}
    components.html(
        live_panel.build_panel_html(_feed_port(), palette, FACE_COLORS, AXIS_COLOR),
        height=height,
    )


def _render_mission_control(state, noisy, fdir, stack_result, active_faults, stack_needed):
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
        vc  = RED if is_alert else TEXT
        bc  = f"{RED}55" if is_alert else BORDER
        top = f"border-top:3px solid {RED};" if is_alert else f"border-top:3px solid {BORDER};"
        with scols[i]:
            st.markdown(
                f'<div style="background:{WHITE};border:1px solid {bc};{top}border-radius:10px;'
                f'padding:10px 6px;text-align:center;box-shadow:0 1px 5px rgba(0,0,0,0.06);">'
                f'<div style="font-size:0.6em;font-weight:700;color:{TEXT_D};letter-spacing:0.09em;'
                f'margin-bottom:5px;">{label.upper()}</div>'
                f'<div style="font-size:1.0em;font-weight:700;color:{vc};font-family:ui-monospace,monospace;">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── G0 SENTINEL STRIP (continuous watch) ─────────────────────────────────
    _render_sentinel_strip()

    # Hardware-link expander is advanced-only — a demo audience does not need
    # to see the serial port picker or the raw MPU6050 traces.
    if not _is_simplified():
        with st.expander("⚙  Hardware Link — Board Connection & Sensor Traces", expanded=False):
            _render_hardware_expander()

    st.markdown("---")

    if not fault_active:
        st.markdown(
            f'<div style="color:{TEXT_D};font-size:0.7em;font-weight:700;letter-spacing:0.12em;margin-bottom:8px;">'
            f'INJECT FAULT EVENTS — SELECT ONE OR MORE SUBSYSTEM FAILURES:</div>',
            unsafe_allow_html=True,
        )
        _fault_buttons(active_faults, "btn")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _, mcol, _ = st.columns([0.25, 3.5, 0.25])
        with mcol:
            _live_model()
            st.markdown(
                f'<div style="text-align:center;color:{TEXT_D};font-size:0.76em;margin-top:2px;">'
                f'Select one or more faults above to trigger FDIR recovery and the Gemma reasoning stack</div>',
                unsafe_allow_html=True,
            )
        return

    # ── FAULT ACTIVE ─────────────────────────────────────────────────────────
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

    left_col, mid_col, right_col = st.columns([1, 1.9, 1.6])

    with left_col:
        _render_fdir_column(state, fdir, active_faults)

    with mid_col:
        _section_label("SPACECRAFT STATE")
        _live_model(height=440)
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

    with right_col:
        _render_stack_column(state, fdir, stack_result, stack_needed)


def _render_fdir_column(state, fdir, active_faults):
    _section_label("FDIR AUTONOMOUS RECOVERY")

    for fid in active_faults:
        fd = fault_details(state, fid)
        fc = FAULT_COLOR.get(fid, AMBER)
        st.markdown(_card(
            f'<div style="color:{fc};font-weight:700;font-size:0.83em;margin-bottom:4px;">⚡ {fd.get("label","").upper()}</div>'
            f'<div style="color:{TEXT_M};font-size:0.75em;line-height:1.45;">{fd.get("description","")}</div>',
            left_color=fc,
        ), unsafe_allow_html=True)

    if fdir and fdir.triggered:
        # Simplified view: only show detection + isolation lines. The full
        # recovery timeline reads like a script log, which is exactly what
        # the Gemma stack replaces — no need to bury the audience in it.
        actions = fdir.actions
        if _is_simplified():
            actions = [a for a in actions if a.phase in ("detection", "isolation", "monitoring")]
        for a in actions:
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


def _render_stack_column(state, fdir, stack_result: StackResult | None, stack_needed: bool):
    _section_label("GEMMA REASONING STACK")

    if stack_needed and stack_result is None:
        with st.spinner("Five-tier stack running — G1 ∥ G4, then G2, then G3…"):
            _run_stack_now(state, fdir)
        st.rerun()

    if stack_result is None:
        st.markdown(_card(
            f'<div style="color:{TEXT_D};font-size:0.8em;text-align:center;padding:20px 0;">'
            f'Awaiting five-tier stack pass</div>',
        ), unsafe_allow_html=True)
        return

    simplified = _is_simplified()

    # Live/cache status line + pipeline strip
    if st.session_state.get("gemma_live"):
        st.markdown(
            f'<div style="margin-bottom:6px;">{_badge(f"◆ LIVE — {MODEL_NAME}", GREEN, sm=True)}</div>',
            unsafe_allow_html=True,
        )
    else:
        reason = st.session_state.get("gemma_error")
        label = "OFFLINE — LIVE CALLS DISABLED" if not USE_LIVE_GEMMA else "SOME TIERS FELL BACK"
        st.markdown(
            f'<div style="margin-bottom:6px;">{_badge("⛃ " + label, AMBER, sm=True)}</div>'
            + (f'<div style="font-size:0.68em;color:{TEXT_D};line-height:1.45;'
               f'margin:-2px 0 6px;">{reason}</div>' if reason and not simplified else ""),
            unsafe_allow_html=True,
        )

    stack_ui.render_pipeline_strip(stack_result.traces, stack_result.wall_time_s,
                                    stack_result.plan_iterations, simplified=simplified)

    # G1: competing hypotheses
    stack_ui.render_diagnosis(stack_result.diagnosis, simplified=simplified)

    # G2 plan + gate outcomes
    if stack_result.plan and stack_result.validation is not None:
        stack_ui.render_plan_and_gates(stack_result.plan, stack_result.validation,
                                        stack_result.verdict, simplified=simplified)

    # G3 adjudicator verdict
    stack_ui.render_adjudicator(stack_result.verdict)

    # Execute / commit control
    approved = bool(
        stack_result.validation and stack_result.validation.approved
        and stack_result.verdict and stack_result.verdict.approved
    )
    committed = st.session_state.get("plan_committed", False)
    button_label = "▶ EXECUTE APPROVED PLAN" if not committed else "✓ PLAN COMMITTED"
    if approved and not committed:
        if st.button(button_label, key="commit_plan", use_container_width=True, type="primary"):
            _commit_plan()
            st.rerun()

    # Before / after
    if stack_result.projected_state and st.session_state.get("pre_plan_state"):
        stack_ui.render_before_after(st.session_state.pre_plan_state, stack_result.projected_state)


# ── G0 sentinel strip (live only, throttled) ────────────────────────────────

SENTINEL_INTERVAL_S = 4.0     # cheap enough to run frequently, deliberately slower than 1 Hz
                              # to keep API budget honest during an unattended demo.


def _render_sentinel_strip():
    link = _serial_link()
    if not link.is_live():
        # Only render the sentinel when there is actually a stream to watch —
        # a "SENTINEL: nominal" chip on a dead feed would be misleading.
        return

    now = time.time()
    last_run = st.session_state.get("sentinel_last_run", 0.0)
    if now - last_run > SENTINEL_INTERVAL_S:
        samples = link.history()
        featurised = featurize_spacecraft_state(st.session_state.spacecraft, samples=samples)
        result = run_sentinel(featurised.get("hardware_features", featurised))
        st.session_state.sentinel_verdict = result.payload
        st.session_state.sentinel_ok = result.trace.ok and result.trace.is_live
        st.session_state.sentinel_latency = result.trace.latency_s
        st.session_state.sentinel_last_run = now

    verdict = st.session_state.get("sentinel_verdict")
    stack_ui.render_sentinel_strip(verdict,
                                    is_live=st.session_state.get("sentinel_ok", False),
                                    latency_s=st.session_state.get("sentinel_latency", 0.0))


# ── Bottom row: G4 capsule + benchmark ──────────────────────────────────────

def _render_bottom_row(state, stack_result: StackResult | None):
    with st.expander("🗂  G4 · Evidence Capsule — PARALLAX knapsack vs. naive baseline",
                     expanded=False):
        _render_capsule_expander(state, stack_result)

    with st.expander("📊 Benchmark harness — five-tier stack vs. bare FDIR across scenarios",
                     expanded=False):
        _render_benchmark_expander()


def _render_capsule_expander(state, stack_result: StackResult | None):
    if stack_result is None or stack_result.packing is None:
        st.markdown(
            f'<div style="color:{TEXT_D};font-size:0.8em;">'
            f'Inject a fault and run the reasoning stack to populate the capsule.</div>',
            unsafe_allow_html=True,
        )
        return

    from parallax.benchmark import _synthetic_evidence
    fids = list(state.active_faults)
    evidence = _synthetic_evidence(fids)

    ccol, _ = st.columns([1, 3])
    with ccol:
        budget_kb = st.slider("Downlink budget (kB)", min_value=8, max_value=60,
                              value=st.session_state.get("capsule_budget_kb", 25),
                              step=1, key="capsule_budget_slider")
        st.session_state["capsule_budget_kb"] = budget_kb

    hypotheses = stack_result.diagnosis.hypotheses if stack_result.diagnosis else []
    stack_ui.render_capsule_panel(stack_result.packing, hypotheses, evidence,
                                   budget_bytes=budget_kb * 1024)


def _render_benchmark_expander():
    ccol, statecol = st.columns([1, 3])
    with ccol:
        run = st.button("▶ Run benchmark", key="bench_run", use_container_width=True)
    with statecol:
        st.markdown(
            f'<div style="color:{TEXT_D};font-size:0.75em;padding-top:10px;line-height:1.5;">'
            f'{len(SCENARIOS)} scenarios — 6 single faults + 3 compound "recovery-scripts-fight-each-other" cases. '
            f'Each runs through bare FDIR and through the five-tier stack.'
            f'</div>',
            unsafe_allow_html=True,
        )

    if run:
        with st.spinner("Running benchmark — this makes real Gemma calls…"):
            report = run_all(SCENARIOS)
        st.session_state.bench_report = report

    report = st.session_state.get("bench_report")
    if report:
        stack_ui.render_benchmark_report(report)


# ── Hardware infrastructure (unchanged) ─────────────────────────────────────

@st.cache_resource
def _serial_link() -> hardware.SerialLink:
    return hardware.SerialLink()


@st.cache_resource
def _feed_port() -> int:
    return telemetry_server.start_server(_serial_link())


FACE_COLORS = {
    "+X": BLUE,   "-X": "#bcd0fa",
    "+Y": AMBER,  "-Y": "#f8dfb8",
    "+Z": PURPLE, "-Z": "#d9c6fb",
}

CHART_MAX_POINTS = 400


def _timeseries(samples, series, y_title, zero_line=False):
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


def _render_hardware_expander():
    link = _serial_link()

    if not hardware.SERIAL_AVAILABLE:
        st.markdown(_card(
            f'<div style="color:{RED};font-weight:700;font-size:0.82em;margin-bottom:4px;">PYSERIAL NOT INSTALLED</div>'
            f'<div style="color:{TEXT_M};font-size:0.75em;">Run <code>pip install pyserial</code> and restart the dashboard.</div>',
            left_color=RED,
        ), unsafe_allow_html=True)
        return

    ports = hardware.list_ports()
    port_ids = [device for device, _ in ports]

    c1, c2, c3, c4 = st.columns([2.5, 1, 1, 2.4])
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
    with c2:
        if st.button("▶ Connect", key="hw_connect", use_container_width=True, disabled=port is None):
            link.start(port, hardware.BAUD_RATE)
            time.sleep(1.2)
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
            f'<div style="color:{TEXT_D};font-size:0.78em;margin-top:8px;line-height:1.6;">'
            f'Select the board\'s port and press <strong>Connect</strong>. '
            f'Close the Arduino IDE Serial Monitor first — only one program can hold the port at a time.<br>'
            f'Expected firmware: <code>parallax_handle</code> (MPU6050 gyro + accel, DS18B20, HC-SR04) '
            f'at {hardware.BAUD_RATE} baud.</div>',
            unsafe_allow_html=True,
        )
        return

    _render_traces()


@st.fragment(run_every=0.8)
def _render_traces():
    link = _serial_link()
    latest = link.latest()
    samples = link.history()
    if latest is None:
        return

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
