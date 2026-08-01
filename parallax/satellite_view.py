"""State-synchronised Plotly model of the Asteria-7 spacecraft."""

from __future__ import annotations

import math
import plotly.graph_objects as go

from parallax.spacecraft import SpacecraftState


STATUS_COLORS = {
    "nominal": "#059669",
    "degraded": "#d97706",
    "failed": "#dc2626",
    "recovering": "#2563eb",
    "standby": "#94a3b8",
}


def _status(state: SpacecraftState, component: str, subsystem: str, default: str = "nominal") -> str:
    return state.component_states.get(component, state.subsystem_health.get(subsystem, default))


def component_states(state: SpacecraftState) -> list[dict]:
    """Describe every rendered component using the dashboard's live state."""
    power_status = state.subsystem_health.get("Power", "nominal")
    return [
        {"name": "Solar array +X", "status": state.component_states.get("Solar array +X", "degraded" if state.solar_efficiency_pct < 95 else power_status), "detail": f"{state.solar_output_w:.0f} W · {state.solar_efficiency_pct:.0f}% efficiency"},
        {"name": "Solar array -X", "status": state.component_states.get("Solar array -X", "degraded" if state.solar_efficiency_pct < 95 else power_status), "detail": f"{state.solar_output_w:.0f} W · {state.solar_efficiency_pct:.0f}% efficiency"},
        {"name": "Power conditioning unit", "status": _status(state, "Power conditioning unit", "Power"), "detail": f"{state.bus_voltage_v:.2f} V bus · {state.pcu_temp_c:.1f}°C"},
        {"name": "Radiator panels", "status": _status(state, "Radiator panels", "Thermal"), "detail": "ACTIVE" if state.radiator_active else "STANDBY"},
        {"name": "Reaction-wheel assembly", "status": _status(state, "Reaction-wheel assembly", "ADCS"), "detail": f"{state.rw_speed_rpm:.0f} rpm · {state.adcs_mode.replace('_', ' ')}"},
        {"name": "High-gain antenna", "status": state.component_states.get("High-gain antenna", "nominal" if state.antenna_mode == "hga" else "failed"), "detail": state.antenna_mode.upper().replace("_", " ")},
        {"name": "Low-gain antenna", "status": state.component_states.get("Low-gain antenna", "nominal" if state.antenna_mode.startswith("lga") else "standby"), "detail": f"{state.data_rate_mbps:.3f} Mbps"},
        {"name": "Spectrometer", "status": state.component_states.get("Spectrometer", "failed" if state.spectrometer_status == "failed" else _status(state, "Spectrometer", "Science")), "detail": state.spectrometer_status.upper()},
        {"name": "Camera", "status": state.component_states.get("Camera", "standby" if state.camera_status == "standby" else _status(state, "Camera", "Science")), "detail": state.camera_status.upper()},
        {"name": "Thrusters", "status": state.component_states.get("Thrusters", "failed" if state.thruster_status != "nominal" else _status(state, "Thrusters", "Propulsion")), "detail": state.thruster_status.upper()},
    ]


def _cuboid(name: str, center: tuple[float, float, float], size: tuple[float, float, float], color: str, hover: str) -> go.Mesh3d:
    cx, cy, cz = center
    sx, sy, sz = (dimension / 2 for dimension in size)
    x = [cx - sx, cx + sx, cx + sx, cx - sx, cx - sx, cx + sx, cx + sx, cx - sx]
    y = [cy - sy, cy - sy, cy + sy, cy + sy, cy - sy, cy - sy, cy + sy, cy + sy]
    z = [cz - sz, cz - sz, cz - sz, cz - sz, cz + sz, cz + sz, cz + sz, cz + sz]
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 7],
        j=[1, 2, 3, 5, 6, 3, 5, 6, 6, 7, 7, 4],
        k=[2, 3, 4, 6, 2, 6, 6, 7, 2, 3, 3, 0],
        color=color, opacity=0.95, flatshading=True, name=name,
        hovertemplate=f"<b>{name}</b><br>{hover}<extra></extra>",
        showscale=False,
    )


def _status_hover(component: dict) -> str:
    return f"Status: {component['status'].upper()}<br>{component['detail']}"


def build_satellite_figure(state: SpacecraftState) -> go.Figure:
    """Render a lightweight 3D satellite whose component colours track state."""
    components = {item["name"]: item for item in component_states(state)}
    color = lambda name: STATUS_COLORS.get(components[name]["status"], "#64748b")
    hover = lambda name: _status_hover(components[name])

    fig = go.Figure()
    # Central bus, panels, radiators and payloads.
    fig.add_trace(_cuboid("Spacecraft bus", (0, 0, 0), (2.1, 1.7, 1.4), "#a8b8d0", "Asteria-7 spacecraft bus"))
    fig.add_trace(_cuboid("Solar array +X", (3.3, 0, 0), (4.2, 0.12, 1.25), color("Solar array +X"), hover("Solar array +X")))
    fig.add_trace(_cuboid("Solar array -X", (-3.3, 0, 0), (4.2, 0.12, 1.25), color("Solar array -X"), hover("Solar array -X")))
    fig.add_trace(_cuboid("Power conditioning unit", (0, -0.95, 0), (0.85, 0.32, 0.7), color("Power conditioning unit"), hover("Power conditioning unit")))
    fig.add_trace(_cuboid("Radiator panels", (0, 0.98, 0), (1.7, 0.08, 1.05), color("Radiator panels"), hover("Radiator panels")))
    fig.add_trace(_cuboid("Spectrometer", (0.72, 0, 0.93), (0.7, 0.65, 0.35), color("Spectrometer"), hover("Spectrometer")))
    fig.add_trace(_cuboid("Camera", (-0.72, 0, 0.93), (0.48, 0.5, 0.32), color("Camera"), hover("Camera")))

    # Antennae and thrusters use cones so their orientations remain legible.
    fig.add_trace(go.Cone(x=[0], y=[0], z=[1.28], u=[0], v=[0], w=[1.15], anchor="tail", sizemode="absolute", sizeref=0.45,
                            colorscale=[[0, color("High-gain antenna")], [1, color("High-gain antenna")]], showscale=False,
                            name="High-gain antenna", hovertemplate=f"<b>High-gain antenna</b><br>{hover('High-gain antenna')}<extra></extra>"))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[-0.65], u=[0], v=[0], w=[-0.8], anchor="tail", sizemode="absolute", sizeref=0.28,
                            colorscale=[[0, color("Low-gain antenna")], [1, color("Low-gain antenna")]], showscale=False,
                            name="Low-gain antenna", hovertemplate=f"<b>Low-gain antenna</b><br>{hover('Low-gain antenna')}<extra></extra>"))
    for index, (x, y) in enumerate(((0.72, 0.52), (-0.72, 0.52), (0.72, -0.52), (-0.72, -0.52)), start=1):
        fig.add_trace(go.Cone(x=[x], y=[y], z=[-0.7], u=[0], v=[0], w=[-0.52], anchor="tail", sizemode="absolute", sizeref=0.23,
                                colorscale=[[0, color("Thrusters")], [1, color("Thrusters")]], showscale=False,
                                name=f"Thruster {index}", hovertemplate=f"<b>Thrusters</b><br>{hover('Thrusters')}<extra></extra>"))

    # Four small rings represent the reaction wheel cluster.
    wheel = components["Reaction-wheel assembly"]
    for center_x, center_y in ((-0.42, -0.42), (0.42, -0.42), (-0.42, 0.42), (0.42, 0.42)):
        angles = [index * (2 * math.pi / 24) for index in range(25)]
        fig.add_trace(go.Scatter3d(
            x=[center_x + .20 * math.cos(angle) for angle in angles],
            y=[center_y + .20 * math.sin(angle) for angle in angles], z=[0.74] * len(angles),
            mode="lines", line={"color": STATUS_COLORS.get(wheel["status"], "#64748b"), "width": 5},
            name="Reaction-wheel assembly", hovertemplate=f"<b>Reaction-wheel assembly</b><br>{_status_hover(wheel)}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=360, margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": -1.9, "z": 1.25}},
        },
    )
    return fig
