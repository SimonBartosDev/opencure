#!/usr/bin/env python3
"""
Send OpenCure outreach emails via Gmail SMTP.

Usage:
    python scripts/send_outreach.py                          # Send all
    python scripts/send_outreach.py --dry-run                # Preview only
    python scripts/send_outreach.py --limit 5                # Send first 5
    python scripts/send_outreach.py --disease "Malaria"      # One disease only
"""

from __future__ import annotations

import argparse
import csv
import smtplib
import ssl
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "opencure.research@gmail.com"
SENDER_NAME = "OpenCure Research"
CSV_PATH = "researchers_outreach.csv"

# Delay between emails (seconds) to avoid rate limits
SEND_DELAY = 2.0


def load_emails(csv_path: str, disease_filter: str = None) -> list[dict]:
    """Load outreach entries from CSV, filtering to those with emails."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("email") or "@" not in row["email"]:
                continue
            if disease_filter and disease_filter.lower() not in row.get("disease", "").lower():
                continue
            rows.append(row)
    return rows


def parse_draft(draft: str) -> tuple[str, str]:
    """Extract subject and body from an email draft string."""
    lines = draft.strip().split("\n")
    subject = ""
    body_lines = []

    if lines and lines[0].startswith("Subject:"):
        subject = lines[0].replace("Subject:", "").strip()
        body_lines = lines[1:]
    else:
        subject = "OpenCure AI Drug Repurposing Predictions"
        body_lines = lines

    body = "\n".join(body_lines).strip()
    return subject, body


def send_email(
    smtp_conn: smtplib.SMTP,
    to_email: str,
    subject: str,
    body: str,
) -> bool:
    """Send a single email. Returns True on success."""
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg["Reply-To"] = SENDER_EMAIL

        msg.attach(MIMEText(body, "plain", "utf-8"))

        smtp_conn.send_message(msg)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to send to {to_email}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Send OpenCure outreach emails")
    parser.add_argument("--app-password", type=str, required=True, help="Gmail App Password")
    parser.add_argument("--dry-run", action="store_true", help="Preview without sending")
    parser.add_argument("--limit", type=int, default=0, help="Max emails to send (0=all)")
    parser.add_argument("--disease", type=str, help="Filter to one disease")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to CSV")
    args = parser.parse_args()

    emails = load_emails(args.csv, args.disease)
    if args.limit > 0:
        emails = emails[:args.limit]

    print(f"OpenCure Outreach Mailer")
    print(f"  Emails to send: {len(emails)}")
    print(f"  Sender: {SENDER_EMAIL}")
    print(f"  Dry run: {args.dry_run}")
    print()

    if not emails:
        print("No emails to send.")
        return

    if args.dry_run:
        for e in emails[:3]:
            subject, body = parse_draft(e["email_draft"])
            print(f"  TO: {e['email']}")
            print(f"  SUBJECT: {subject}")
            print(f"  BODY (first 200 chars): {body[:200]}...")
            print()
        print(f"  ... and {len(emails) - min(3, len(emails))} more")
        return

    # Connect to Gmail SMTP
    print("Connecting to Gmail SMTP...")
    context = ssl.create_default_context()
    smtp = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    smtp.starttls(context=context)
    smtp.login(SENDER_EMAIL, args.app_password.replace(" ", ""))
    print("Connected.\n")

    sent = 0
    failed = 0
    for i, entry in enumerate(emails):
        to = entry["email"]
        subject, body = parse_draft(entry["email_draft"])
        disease = entry.get("disease", "")
        name = entry.get("researcher_name", "")

        print(f"  [{i+1}/{len(emails)}] {name} ({disease}) → {to}")

        if send_email(smtp, to, subject, body):
            sent += 1
        else:
            failed += 1

        time.sleep(SEND_DELAY)

    smtp.quit()

    print(f"\nDone: {sent} sent, {failed} failed out of {len(emails)} total.")


if __name__ == "__main__":
    main()
