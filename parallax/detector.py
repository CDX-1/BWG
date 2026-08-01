import pandas as pd


def detect_event(df: pd.DataFrame) -> tuple[bool, dict]:
    spectrum_alert = df["spectrum"].max() > 15
    radiation_alert = df["radiation"].max() > 38
    checksum_alert = df["checksum_errors"].sum() > 0

    triggered = spectrum_alert and (radiation_alert or checksum_alert)

    details = {
        "spectrum_alert": spectrum_alert,
        "radiation_alert": radiation_alert,
        "checksum_alert": checksum_alert,
        "spectrum_peak": float(df["spectrum"].max()),
        "radiation_peak": float(df["radiation"].max()),
        "checksum_errors": int(df["checksum_errors"].sum()),
        "anomaly_index": int(df["spectrum"].idxmax()),
    }

    return triggered, details
