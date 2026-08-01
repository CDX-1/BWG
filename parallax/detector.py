import pandas as pd


def detect_event(df: pd.DataFrame) -> tuple[bool, dict]:
    cols = set(df.columns)

    if "spectrum" in cols:
        spectrum_alert = df["spectrum"].max() > 15
        radiation_alert = df["radiation"].max() > 38
        checksum_alert = df["checksum_errors"].sum() > 0
        triggered = spectrum_alert and (radiation_alert or checksum_alert)
        anomaly_index = int(df["spectrum"].idxmax())

    elif "solar_output" in cols:
        baseline = df["solar_output"][:30].mean()
        power_alert = df["solar_output"].min() < baseline * 0.75
        thermal_alert = df["pcu_temp"].max() > 47
        triggered = power_alert and thermal_alert
        anomaly_index = int(df["solar_output"].idxmin())

    else:
        triggered = False
        anomaly_index = 60

    return triggered, {"anomaly_index": anomaly_index}
