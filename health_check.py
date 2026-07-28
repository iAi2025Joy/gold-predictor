"""
health_check.py
==================

Lightweight reliability check, run on a schedule, to catch failures that
would otherwise go unnoticed until a user (or you) happens to test the
site by chance -- which is exactly what happened twice with two
accidental goldPrediction.js deletions during recent edits.

HOW THIS ALERTS YOU: this script intentionally EXITS WITH A NON-ZERO
STATUS when something is wrong. GitHub Actions automatically marks a
failed workflow run red in the Actions tab, and (if your notification
settings allow it -- see the setup note at the bottom of this file)
emails the repo owner when a SCHEDULED workflow run fails. That's a
genuinely free, zero-setup alerting channel -- no Slack/email
integration code needed, just an honest exit code.

WHAT THIS CHECKS:
1. Is the live chatbot backend actually responding? (catches the exact
   "file got deleted, backend crashes on startup" failure mode seen
   twice already)
2. Is the gold prediction data fresh? (catches the hourly job silently
   failing or stopping without anyone noticing)
3. Same freshness check for the oil prediction data.

WHAT THIS DOES NOT DO: it can't fix or restart anything -- GitHub
Actions has no ability to reach into an external Render service and
redeploy it. It only makes a failure VISIBLE (a red X + an email)
instead of silent, which is the actual gap this is closing.
"""

import sys
import json
from datetime import datetime, timezone

import requests

BACKEND_HEALTH_URL = "https://ai-chat-backend-3-g573.onrender.com/"
GOLD_PREDICTION_FILE = "gold_prediction_latest.json"
OIL_PREDICTION_FILE = "oil_prediction_latest.json"
DXY_PREDICTION_FILE = "dxy_prediction_latest.json"
STALE_THRESHOLD_HOURS = 4  # hourly job runs every hour -- this allows a couple of missed/delayed runs before actually alerting, to avoid false alarms
DXY_STALE_THRESHOLD_HOURS = 10  # DXY updates every 6 hours, not hourly -- allow one missed cycle


def check_backend_health():
    """Render's free tier spins the service down after ~15 minutes of
    inactivity, so the FIRST request after idle time can take up to
    ~50 seconds to wake it back up -- give it real time and one retry
    before honestly calling it down, so a slow cold-start doesn't get
    misreported as an outage."""
    last_error = None
    for attempt in range(2):
        try:
            resp = requests.get(BACKEND_HEALTH_URL, timeout=60)
            if resp.status_code == 200:
                return True, "Backend responded 200 OK."
            last_error = f"HTTP {resp.status_code}"
        except Exception as err:
            last_error = str(err)
    return False, f"Backend did not respond with 200 OK after 2 attempts (last error: {last_error})."


def check_freshness(filename, label, threshold_hours=STALE_THRESHOLD_HOURS):
    try:
        with open(filename) as f:
            data = json.load(f)
    except FileNotFoundError:
        return False, f"{label}: file '{filename}' not found in repo."
    except json.JSONDecodeError as err:
        return False, f"{label}: file '{filename}' exists but is not valid JSON ({err})."

    updated_at = data.get("updated_at")
    if not updated_at:
        return False, f"{label}: no 'updated_at' field found in {filename}."

    try:
        updated_dt = datetime.fromisoformat(updated_at)
    except ValueError:
        return False, f"{label}: could not parse 'updated_at' timestamp '{updated_at}'."

    age_hours = (datetime.now(timezone.utc) - updated_dt).total_seconds() / 3600
    if age_hours > threshold_hours:
        return False, (
            f"{label}: data is {age_hours:.1f} hours old (threshold: {threshold_hours}h) -- "
            f"the update job may have stopped running or is failing."
        )
    return True, f"{label}: fresh, last updated {age_hours:.1f} hours ago."


def main():
    checks = [
        check_backend_health(),
        check_freshness(GOLD_PREDICTION_FILE, "Gold prediction"),
        check_freshness(OIL_PREDICTION_FILE, "Oil prediction"),
        check_freshness(DXY_PREDICTION_FILE, "DXY prediction", threshold_hours=DXY_STALE_THRESHOLD_HOURS),
    ]

    all_ok = True
    print("=== Reliability Health Check ===")
    for ok, message in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            all_ok = False

    if not all_ok:
        print(
            "\nOne or more checks failed. This workflow will now exit with a "
            "non-zero status so GitHub Actions marks it red and (if your "
            "notification settings allow it) emails you about the failed "
            "scheduled run."
        )
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
