import time
from dataclasses import dataclass, field
from parallax.spacecraft import SpacecraftState, fault_details


@dataclass
class FDIRAction:
    timestamp: str
    fault_id: str
    phase: str  # "detection", "isolation", "recovery"
    message: str
    outcome: str  # "success", "partial", "in_progress", "failed"


@dataclass
class FDIRReport:
    triggered: bool
    active_faults: list[str]
    actions: list[FDIRAction]
    post_recovery_state: SpacecraftState
    mission_safe: bool
    summary: str


FAULT_DETECTION_RULES = {
    "solar_string_loss": lambda s: s.solar_output_w < 600,
    "pcu_fault": lambda s: s.bus_voltage_v < 26.0 or s.pcu_temp_c > 55,
    "reaction_wheel_fault": lambda s: s.rw_speed_rpm < 500 or s.attitude_error_arcsec > 5.0,
    "spectrometer_fault": lambda s: s.spectrometer_status == "failed" or s.spectrometer_output is None,
    "comms_dropout": lambda s: s.signal_strength_dbm < -122,
    "thermal_runaway": lambda s: s.pcu_temp_c > 60 or s.bus_temp_c > 36,
}

FAULT_ISOLATION = {
    "solar_string_loss": "Isolated to solar array string 2. Strings 1, 3, 4 nominal. Battery discharge rate elevated.",
    "pcu_fault": "Isolated to PCU module B. Backup PCU-A available. Downstream voltage regulators affected.",
    "reaction_wheel_fault": "Isolated to reaction wheel 2. RW 1, 3, 4 nominal. Magnetic torquers available as backup.",
    "spectrometer_fault": "Isolated to spectrometer detector array. Camera and radiation sensor unaffected.",
    "comms_dropout": "Isolated to HGA pointing actuator. Low-gain antenna LGA-2 available at reduced data rate.",
    "thermal_runaway": "Isolated to PCU thermal zone. Heat spreading to instrument bay. Active cooling required.",
}

FAULT_RECOVERY = {
    "solar_string_loss": [
        ("Battery conservation mode activated. Non-essential loads shed.", "success"),
        ("Science instrument power reduced 40%.", "success"),
        ("Solar array rotation angle adjusted +2.3° toward sun.", "success"),
        ("Power budget restabilised at 68% nominal. Monitoring.", "partial"),
    ],
    "pcu_fault": [
        ("Switching to backup PCU-A. 800ms switchover in progress.", "in_progress"),
        ("Backup PCU-A online. Bus voltage restabilising.", "success"),
        ("PCU-B isolated and powered down. Thermal load reduced.", "success"),
        ("Power subsystem operating on backup. Full diagnostic scheduled.", "partial"),
    ],
    "reaction_wheel_fault": [
        ("Magnetic torquer attitude control engaged as backup.", "success"),
        ("RW-2 spin-down commanded. Bearing cooldown initiated.", "success"),
        ("Attitude error correcting. Sun-pointing recovery sequence active.", "in_progress"),
        ("ADCS stabilised at degraded precision (±4.2 arcsec). Science pointing limited.", "partial"),
    ],
    "spectrometer_fault": [
        ("Power cycle initiated on spectrometer detector array.", "success"),
        ("Instrument buffers flushed. Diagnostic sequence running.", "in_progress"),
        ("Detector array unresponsive. Instrument placed in safe mode.", "partial"),
        ("Science acquisition suspended. Camera and radiation sensor remain active.", "success"),
    ],
    "comms_dropout": [
        ("HGA actuator fault confirmed. Switching to LGA-2 backup.", "success"),
        ("LGA-2 acquired Earth signal at -118.4 dBm.", "success"),
        ("Data rate reduced to 115 kbps. Downlink queue prioritised.", "success"),
        ("Comms operating in contingency mode. HGA recovery scheduled.", "partial"),
    ],
    "thermal_runaway": [
        ("Emergency thermal control activated. Radiator panels deployed.", "success"),
        ("PCU computational load reduced 30%. Excess heat output decreasing.", "success"),
        ("Instrument bay heaters disabled. Passive cooling active.", "success"),
        ("PCU temperature stabilising at 52°C. Thermal margin restored.", "partial"),
    ],
}


def run_fdir(state: SpacecraftState) -> FDIRReport:
    """Run the full FDIR cycle: detect → isolate → recover."""
    actions = []
    # An injected custom anomaly is an operator-observed event, so it is
    # processed directly.  When no event is supplied, preserve the detector's
    # autonomous threshold-based behaviour for incoming telemetry.
    detected = list(dict.fromkeys(state.active_faults))

    # Detection pass
    if not detected:
        for fault_id, rule in FAULT_DETECTION_RULES.items():
            if rule(state):
                detected.append(fault_id)

    for fault_id in detected:
        details = fault_details(state, fault_id)
        if fault_id in state.fault_metadata:
            message = f"EVENT DETECTED: {details['label'].upper()}"
        else:
            message = f"FAULT DETECTED: {fault_id.upper().replace('_', ' ')}"
        actions.append(FDIRAction(
            timestamp="T+00:01",
            fault_id=fault_id,
            phase="detection",
            message=message,
            outcome="success",
        ))

    if not detected:
        return FDIRReport(
            triggered=False,
            active_faults=[],
            actions=[FDIRAction("T+00:00", "", "monitoring", "All systems nominal. FDIR monitoring active.", "success")],
            post_recovery_state=state,
            mission_safe=True,
            summary="All subsystems operating within nominal parameters.",
        )

    # Isolation pass
    for fault_id in detected:
        details = fault_details(state, fault_id)
        if fault_id in state.fault_metadata:
            targets = ", ".join(details.get("subsystems", []))
            isolation = (
                f"Operator-defined event affecting {targets}. "
                f"Preserving raw telemetry and holding affected subsystem(s) at "
                f"{details.get('severity', 'warning').upper()} severity pending analysis."
            )
        else:
            isolation = FAULT_ISOLATION[fault_id]
        actions.append(FDIRAction(
            timestamp="T+00:02",
            fault_id=fault_id,
            phase="isolation",
            message=f"ISOLATION: {isolation}",
            outcome="success",
        ))

    # Recovery pass
    for fault_id in detected:
        if fault_id in state.fault_metadata:
            details = fault_details(state, fault_id)
            steps = [
                ("Raw telemetry snapshot preserved for ground review.", "success"),
                (f"Gemma assessment requested for: {details['description']}", "in_progress"),
                ("No irreversible recovery command issued without validated procedure.", "partial"),
            ]
        else:
            steps = FAULT_RECOVERY.get(fault_id, [])
        for i, (msg, outcome) in enumerate(steps):
            actions.append(FDIRAction(
                timestamp=f"T+00:0{3 + i}",
                fault_id=fault_id,
                phase="recovery",
                message=f"RECOVERY: {msg}",
                outcome=outcome,
            ))

    # Determine mission safety
    critical_faults = {"pcu_fault", "thermal_runaway"}
    has_critical_custom_event = any(
        state.fault_metadata.get(fault_id, {}).get("severity") == "critical"
        for fault_id in detected
    )
    mission_safe = not (any(f in critical_faults for f in detected) or has_critical_custom_event) or \
                   all(a.outcome in ("success", "partial") for a in actions if a.phase == "recovery")

    # Build summary
    fault_names = [fault_details(state, f)["label"].upper() for f in detected]
    summary = f"{len(detected)} fault(s) detected and processed: {', '.join(fault_names)}. " \
              f"{'Mission SAFE — operating in contingency mode.' if mission_safe else 'MISSION CRITICAL — immediate ground intervention required.'}"

    return FDIRReport(
        triggered=True,
        active_faults=detected,
        actions=actions,
        post_recovery_state=state,
        mission_safe=mission_safe,
        summary=summary,
    )
