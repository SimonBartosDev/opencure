#!/usr/bin/env python3
"""Review CLI — list and manage agent outputs in the outbox.

Usage:
    python3 agents/review.py              # List all pending drafts
    python3 agents/review.py read <file>  # Read a specific draft
    python3 agents/review.py clear        # Archive all reviewed drafts
    python3 agents/review.py status       # Show agent run history
"""

from __future__ import annotations

import sys
import shutil
from datetime import datetime
from pathlib import Path

OUTBOX = Path(__file__).parent / "outbox"
LOGS = Path(__file__).parent / "logs"
ARCHIVE = Path(__file__).parent / "archive"


def list_drafts():
    """List all pending drafts in the outbox."""
    files = sorted(OUTBOX.glob("*"))
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    if not files:
        print("No pending drafts in outbox.")
        return

    print(f"\n  OpenCure Agent Outbox ({len(files)} drafts)\n")
    print(f"  {'#':<4} {'Date':<12} {'Agent':<25} {'Size':<8}")
    print(f"  {'-'*4} {'-'*12} {'-'*25} {'-'*8}")

    for i, f in enumerate(files, 1):
        parts = f.stem.split("_", 1)
        date = parts[0] if len(parts) > 1 else "unknown"
        agent = parts[1] if len(parts) > 1 else f.stem
        size = f"{f.stat().st_size / 1024:.1f} KB"
        print(f"  {i:<4} {date:<12} {agent:<25} {size:<8}")

    print(f"\n  Read a draft: python3 agents/review.py read {files[0].name}")
    print(f"  Archive all:  python3 agents/review.py clear\n")


def read_draft(filename: str):
    """Display a specific draft."""
    path = OUTBOX / filename
    if not path.exists():
        # Try matching partial name
        matches = list(OUTBOX.glob(f"*{filename}*"))
        if matches:
            path = matches[0]
        else:
            print(f"Not found: {filename}")
            return

    print(path.read_text())


def clear_outbox():
    """Archive all drafts."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    files = [f for f in OUTBOX.glob("*") if f.is_file() and not f.name.startswith(".")]

    if not files:
        print("Outbox is already empty.")
        return

    for f in files:
        dest = ARCHIVE / f.name
        shutil.move(str(f), str(dest))

    print(f"Archived {len(files)} drafts to agents/archive/")


def show_status():
    """Show agent run history."""
    logs = sorted(LOGS.glob("*.log"))

    if not logs:
        print("No agent run history yet.")
        return

    print(f"\n  Agent Run History\n")
    for log in logs:
        agent = log.stem
        lines = log.read_text().strip().split("\n")
        last = lines[-1] if lines else "No runs"
        total = len(lines)
        print(f"  {agent:<25} {total} runs | Last: {last[:80]}")

    print()


def main():
    if len(sys.argv) < 2:
        list_drafts()
        return

    cmd = sys.argv[1]

    if cmd == "read" and len(sys.argv) > 2:
        read_draft(sys.argv[2])
    elif cmd == "clear":
        clear_outbox()
    elif cmd == "status":
        show_status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
