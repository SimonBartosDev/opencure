#!/usr/bin/env python3
"""Report Updater Agent — regenerates all outputs when new data is available.

Checks for new screening results, then regenerates the database, PDF reports,
and Explorer dashboard. Commits and reports what changed.
"""

from __future__ import annotations

import subprocess
import json
from datetime import datetime
from pathlib import Path

from common import PROJECT_ROOT, RESULTS_DIR, DB_PATH, write_outbox, log_run


def count_results() -> int:
    """Count disease result files."""
    return len([f for f in RESULTS_DIR.glob("*.json")
                if not f.name.startswith("opencure_")
                and f.name not in ("screening_summary.json", "novel_candidates.json")])


def get_db_count() -> int:
    """Count candidates in current database."""
    try:
        raw = json.loads(DB_PATH.read_text())
        return len(raw.get("candidates", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def run_script(script_path: str) -> tuple[bool, str]:
    """Run a Python script and return (success, output)."""
    try:
        result = subprocess.run(
            ["python3", script_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Script timed out after 10 minutes"


def run():
    today = datetime.now().strftime("%Y-%m-%d")
    disease_count = count_results()
    old_db_count = get_db_count()

    print(f"[Report Updater] {disease_count} disease results found, {old_db_count} candidates in current DB")

    report = f"# Report Updater — {today}\n\n"
    report += f"Disease result files: {disease_count}\n"
    report += f"Current database candidates: {old_db_count}\n\n"

    # Step 1: Regenerate database
    print("[Report Updater] Regenerating database...")
    ok, output = run_script("experiments/generate_database.py")
    new_db_count = get_db_count()
    report += f"## Database Regeneration\n"
    report += f"- Status: {'OK' if ok else 'FAILED'}\n"
    report += f"- Candidates: {old_db_count} -> {new_db_count} ({new_db_count - old_db_count:+d})\n\n"

    if not ok:
        report += f"Error:\n```\n{output[-500:]}\n```\n\n"

    # Step 2: Regenerate PDF reports
    print("[Report Updater] Regenerating PDF reports...")
    ok2, output2 = run_script("scripts/generate_reports.py")
    report += f"## PDF Reports\n"
    report += f"- Status: {'OK' if ok2 else 'FAILED'}\n\n"

    # Step 3: Regenerate Explorer dashboard
    print("[Report Updater] Regenerating Explorer dashboard...")
    ok3, output3 = run_script("scripts/build_explorer.py")
    report += f"## Explorer Dashboard\n"
    report += f"- Status: {'OK' if ok3 else 'FAILED'}\n\n"

    if new_db_count > old_db_count:
        report += f"**{new_db_count - old_db_count} new candidates added!** Review and push to GitHub.\n\n"
        report += "To push: `git add experiments/results/ reports/ docs/ && git commit -m 'Update results' && git push`\n"
    elif new_db_count == old_db_count:
        report += "No new candidates. Database unchanged.\n"

    report += f"\n*Agent run completed at {datetime.now().isoformat()}*\n"

    path = write_outbox("report_update", report)
    log_run("report_updater", f"DB: {old_db_count}->{new_db_count}, reports: {'OK' if ok2 else 'FAIL'}, dashboard: {'OK' if ok3 else 'FAIL'}")
    print(f"[Report Updater] Done. Report: {path}")


if __name__ == "__main__":
    run()
