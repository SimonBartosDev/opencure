# Prospective validation monitor — scheduling

The monitor (`scripts/prospective_monitor.py`) re-queries PubMed and
ClinicalTrials.gov for evidence published **after** each snapshot date
and updates `data/prospective/summary.json` with rolling precision@K.

It's idempotent; running it more often than monthly is fine but wastes
API calls. The recommended cadence is **monthly on the 1st**.

## macOS (launchd)

Paste the following into `~/Library/LaunchAgents/org.opencure.prospective.plist`
(edit `$REPO_PATH` to the absolute path of this repo):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>         <string>org.opencure.prospective</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/env</string>
        <string>python3</string>
        <string>$REPO_PATH/scripts/prospective_monitor.py</string>
    </array>
    <key>WorkingDirectory</key><string>$REPO_PATH</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Day</key>    <integer>1</integer>
        <key>Hour</key>   <integer>3</integer>
        <key>Minute</key> <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>   <string>$REPO_PATH/logs/prospective.out</string>
    <key>StandardErrorPath</key> <string>$REPO_PATH/logs/prospective.err</string>
    <key>RunAtLoad</key>          <false/>
</dict>
</plist>
```

Then load it:

```bash
launchctl load ~/Library/LaunchAgents/org.opencure.prospective.plist
launchctl list | grep opencure     # confirm it's registered
```

To trigger a one-off run (e.g. after editing):

```bash
launchctl start org.opencure.prospective
tail -f logs/prospective.out
```

To remove: `launchctl unload ~/Library/LaunchAgents/org.opencure.prospective.plist`.

## Linux (cron)

Add to `crontab -e`:

```cron
0 3 1 * * cd /path/to/opencure && /usr/bin/env python3 scripts/prospective_monitor.py >> logs/prospective.log 2>&1
```

## GitHub Actions (runs regardless of your machine)

`.github/workflows/prospective.yml`:

```yaml
name: Prospective validation monitor
on:
  schedule:
    - cron: "0 3 1 * *"     # 03:00 UTC on day 1 of each month
  workflow_dispatch: {}

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: python3 scripts/prospective_monitor.py
      - uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "prospective monitor: monthly precision@K update"
          file_pattern: "data/prospective/**"
```

Uses the same monthly cadence; runs even when your laptop is off; commits
the updated `data/prospective/summary.json` automatically.

## What the output looks like

After one month with the monitor running:

```json
// data/prospective/summary.json
{
  "snapshot_date": "2026-04-18T164121Z",
  "predictions_tracked": 610,
  "predictions_aged_90d": 61,
  "validation_hits": 4,
  "rolling_precision_at_10": 0.067,
  "last_updated": "2026-05-01T03:00:15Z",
  "method": "PubMed + ClinicalTrials.gov re-query after snapshot date",
  "notes": "Hits = new papers or trials published after snapshot date "
           "for (drug, disease) pairs we predicted."
}
```

The first meaningful report lands after predictions age 90 days; see
`docs/methods_paper_draft.md` §4.4.
