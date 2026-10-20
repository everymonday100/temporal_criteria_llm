"""io.py — save/load helpers for JSON and npz caches."""
import json
from pathlib import Path

import numpy as np


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def exists(path):
    return Path(path).exists()