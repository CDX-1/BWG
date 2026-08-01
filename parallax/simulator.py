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
