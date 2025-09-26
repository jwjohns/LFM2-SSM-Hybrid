from __future__ import annotations

import json
import os
import random


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        # Minimal fallback: parse key=value lines
        cfg = {}
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
            elif "=" in line:
                k, v = line.split("=", 1)
            else:
                continue
            cfg[k.strip()] = _coerce(v.strip())
        return cfg


def _coerce(v: str):
    lowers = {"true": True, "false": False}
    if v.lower() in lowers:
        return lowers[v.lower()]
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

