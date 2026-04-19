"""
Structured logging + per-pillar timing for OpenCure.

Single source of truth for logging configuration. Every module imports
`from opencure.log_setup import get_logger` rather than using `print()`
or `logging.getLogger()` directly.

Two output sinks:
  1. Console — human-readable level-color-message at INFO level
  2. File (optional) — JSON lines at DEBUG level, machine-parseable,
     written to experiments/run_<timestamp>.jsonl if OPENCURE_LOG_FILE set

Timing helper: `with timed(logger, "pillar_name"):` logs elapsed seconds.
Metrics accumulator: writes per-pillar timings to data/metrics/timings.csv
for later analysis.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


METRICS_DIR = Path("data/metrics")
_METRICS_FILE = METRICS_DIR / "timings.csv"
_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any .extra fields
        for k, v in record.__dict__.items():
            if k in ("args", "msg", "levelname", "levelno", "pathname",
                     "filename", "module", "exc_info", "exc_text",
                     "stack_info", "lineno", "funcName", "created",
                     "msecs", "relativeCreated", "thread", "threadName",
                     "processName", "process", "name", "getMessage",
                     "taskName"):
                continue
            try:
                json.dumps(v)  # skip unserializable
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        return json.dumps(payload)


class _ColorFormatter(logging.Formatter):
    """Simple color-coded console output."""
    _COLORS = {
        "DEBUG":    "\033[90m",
        "INFO":     "\033[0m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        msg = record.getMessage()
        return f"{color}[{record.levelname:8s}] {record.name}: {msg}{self._RESET}"


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("opencure")
    root.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(os.environ.get("OPENCURE_LOG_LEVEL", "INFO"))
    ch.setFormatter(_ColorFormatter())
    root.addHandler(ch)

    # File (optional)
    file_path = os.environ.get("OPENCURE_LOG_FILE")
    if file_path:
        fh = logging.FileHandler(file_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Module authors pass __name__."""
    _configure_once()
    if not name.startswith("opencure"):
        name = f"opencure.{name}"
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, label: str, **extra):
    """Context manager that logs elapsed time + appends to metrics CSV.

    with timed(logger, "pillar_transe", disease="Malaria"):
        ... heavy work ...

    Emits an INFO log line and a row to data/metrics/timings.csv.
    """
    t0 = time.time()
    logger.debug(f"{label} start", extra={"event": "start", "label": label, **extra})
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info(f"{label}: {dt:.2f}s", extra={"event": "timing", "label": label,
                                                   "elapsed_s": round(dt, 4), **extra})
        _append_timing_metric(label, dt, extra)


def _append_timing_metric(label: str, elapsed_s: float, extra: dict) -> None:
    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        new_file = not _METRICS_FILE.exists()
        with _METRICS_FILE.open("a") as f:
            if new_file:
                f.write("timestamp,label,elapsed_s,context\n")
            ts = datetime.now(timezone.utc).isoformat()
            ctx = json.dumps(extra).replace(",", ";")
            f.write(f"{ts},{label},{elapsed_s:.4f},{ctx}\n")
    except Exception:
        # Never let logging break the pipeline
        pass
