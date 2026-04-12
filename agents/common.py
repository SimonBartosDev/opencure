"""Shared utilities for OpenCure agents."""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTBOX = PROJECT_ROOT / "agents" / "outbox"
LOGS = PROJECT_ROOT / "agents" / "logs"
CONFIG_PATH = PROJECT_ROOT / "agents" / "config.json"
DB_PATH = PROJECT_ROOT / "experiments" / "results" / "opencure_database.json"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def load_predictions(top_n: int = 50) -> list[dict]:
    """Load top N predictions from database, sorted by score."""
    raw = json.loads(DB_PATH.read_text())
    candidates = raw["candidates"]
    candidates.sort(key=lambda c: -c["combined_score"])
    return candidates[:top_n]


def load_all_predictions() -> list[dict]:
    raw = json.loads(DB_PATH.read_text())
    return raw["candidates"]


def write_outbox(agent_name: str, content: str, suffix: str = "md") -> Path:
    """Write agent output to outbox with timestamp."""
    OUTBOX.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = OUTBOX / f"{date_str}_{agent_name}.{suffix}"
    path.write_text(content)
    return path


def log_run(agent_name: str, summary: str):
    """Append a run log entry."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{agent_name}.log"
    entry = f"[{datetime.now().isoformat()}] {summary}\n"
    with open(log_path, "a") as f:
        f.write(entry)


def days_ago(n: int) -> str:
    """Return date string N days ago in YYYY/MM/DD format (PubMed)."""
    d = datetime.now() - timedelta(days=n)
    return d.strftime("%Y/%m/%d")


def get_top_breakthrough_predictions(n: int = 10) -> list[dict]:
    """Get top N breakthrough predictions by score."""
    all_preds = load_all_predictions()
    breakthroughs = [c for c in all_preds if c.get("novelty_level") == "BREAKTHROUGH"]
    breakthroughs.sort(key=lambda c: -c["combined_score"])
    return breakthroughs[:n]


def get_diseases() -> list[str]:
    """Get all screened disease names."""
    all_preds = load_all_predictions()
    return sorted(set(c["disease"] for c in all_preds))
