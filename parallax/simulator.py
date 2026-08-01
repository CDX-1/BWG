import numpy as np
import pandas as pd


def generate_spectrometer_scenario(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, 120)

    spectrum = rng.normal(10, 0.4, len(t))
    radiation = rng.normal(30, 1.2, len(t))
    temperature = rng.normal(18, 0.05, len(t))
    checksum_errors = np.zeros(len(t))

    # Inject anomaly at t=60 (mission time T+04:17:32 maps to index 60)
    spectrum[60:63] += [4, 9, 3]
    radiation[59:64] += [2, 8, 13, 7, 2]
    temperature[61:68] += np.linspace(0.1, 0.8, 7)
    checksum_errors[62] = 2

    return pd.DataFrame({
        "time": t,
        "spectrum": spectrum,
        "radiation": radiation,
        "temperature": temperature,
        "checksum_errors": checksum_errors,
    })


ANOMALY_TIME_INDEX = 60
MISSION_TIME_LABEL = "T+04:17:32"


def generate_power_scenario(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, 120)

    solar_output = rng.normal(850, 5, len(t))       # watts
    battery_soc = np.linspace(94, 92, len(t)) + rng.normal(0, 0.1, len(t))  # % slowly draining
    pcu_temp = rng.normal(41, 0.3, len(t))           # °C
    attitude_error = rng.normal(0, 0.8, len(t))      # arcsec

    # Inject anomaly at t=60: solar drops, battery drains faster, PCU heats up, attitude drifts
    solar_output[60:] -= np.linspace(0, 300, 60)    # progressive drop to ~550W
    battery_soc[60:] -= np.linspace(0, 4, 60)       # faster drain
    pcu_temp[60:75] += np.linspace(0, 8, 15)        # thermal spike then partial recovery
    pcu_temp[75:] += 4
    attitude_error[62:70] += np.linspace(0, 3, 8)   # attitude drift

    return pd.DataFrame({
        "time": t,
        "solar_output": solar_output,
        "battery_soc": battery_soc,
        "pcu_temp": pcu_temp,
        "attitude_error": attitude_error,
    })


POWER_ANOMALY_TIME_INDEX = 60
POWER_MISSION_TIME_LABEL = "T+09:42:15"


SCENARIO_TELEMETRY = {
    "spectrometer_001": (generate_spectrometer_scenario, ANOMALY_TIME_INDEX, MISSION_TIME_LABEL),
    "power_001": (generate_power_scenario, POWER_ANOMALY_TIME_INDEX, POWER_MISSION_TIME_LABEL),
}

SCENARIO_CHART_COLS = {
    "spectrometer_001": [
        ("spectrum",        "#60a5fa", "Spectrometer Intensity"),
        ("radiation",       "#f87171", "Radiation Count"),
        ("temperature",     "#34d399", "Instrument Temperature (°C)"),
        ("checksum_errors", "#fbbf24", "Checksum Errors"),
    ],
    "power_001": [
        ("solar_output",   "#fbbf24", "Solar Array Output (W)"),
        ("battery_soc",    "#34d399", "Battery State of Charge (%)"),
        ("pcu_temp",       "#f87171", "PCU Temperature (°C)"),
        ("attitude_error", "#a78bfa", "Attitude Error (arcsec)"),
    ],
}
