"""
Experiment logging utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    models_dir: Path
    config_path: Path
    metrics_path: Path
    meta_path: Path


def make_run_dir(root: Path, name: str) -> RunPaths:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{ts}_{name}"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        models_dir=models_dir,
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.json",
        meta_path=run_dir / "meta.json",
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    import pandas as pd

    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
