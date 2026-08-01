import json
import os
from parallax.config import SCENARIO_DIR


def load_scenario(scenario_id: str) -> dict:
    path = os.path.join(SCENARIO_DIR, f"{scenario_id}.json")
    with open(path) as f:
        return json.load(f)


def format_bytes(n: int) -> str:
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"
