"""
data_watchdog.py -- warns (and self-heals) when the live data goes stale.

Checks the 'updated_at' time inside each data file in the repo:
  live_gold_ticks.json        (price recorder)         warn after 2 h  -- and RESTARTS the recorder after 45 min
  short_range_latest.json     (10/20/30/60-min ranges) warn after 2 h
  direction_live_record.json  (up/down record)         warn after 2 h
  gold_prediction_latest.json (hourly gold job)        warn after 4 h
Warning = a GitHub Issue titled "Data is stale" (assigned to the repo owner, so GitHub emails them). The issue is
closed automatically when everything is fresh again. Needs the GitHub CLI (`gh`) with GH_TOKEN.
Run: python data_watchdog.py        Self-test: python data_watchdog.py selftest
"""
import os, sys, json, subprocess
from datetime import datetime, timezone, timedelta

CHECKS = [  # (file, warn_after_hours, label)
    ("live_gold_ticks.json", 2, "Live gold price recorder"),
    ("short_range_latest.json", 2, "Short-term price ranges"),
    ("direction_live_record.json", 2, "Up/down direction record"),
    ("gold_prediction_latest.json", 4, "Hourly gold prediction"),
    ("event_calendar.json", 30, "Economic event calendar"),
    ("market_headlines.json", 14, "News headlines"),
]
RESTART_TICKS_AFTER_MIN = 45
ISSUE_TITLE = "Data is stale"
LABEL = "data-stale"

def now_utc():
    return datetime.now(timezone.utc)

def age_minutes(path, now):
    try:
        d = json.load(open(path))
        t = d.get("updated_at") or d.get("generated_at") or d.get("timestamp")
        if not t:
            return None
        ts = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds() / 60
    except Exception:
        return None

def gh(*args, check=False):
    return subprocess.run(["gh", *args], capture_output=True, text=True)

def evaluate(now, base="."):
    problems, ok = [], []
    for f, hours, label in CHECKS:
        a = age_minutes(os.path.join(base, f), now)
        if a is None:
            problems.append(f"- **{label}** (`{f}`): file missing or unreadable.")
        elif a > hours * 60:
            problems.append(f"- **{label}** (`{f}`): last updated {a/60:.1f} hours ago (limit {hours} h).")
        else:
            ok.append(label)
    ticks_age = age_minutes(os.path.join(base, "live_gold_ticks.json"), now)
    return problems, ok, ticks_age

def restart_recorder():
    busy = gh("run", "list", "--workflow", "live_tick_recorder.yml", "--json", "status", "--limit", "10")
    try:
        statuses = [r["status"] for r in json.loads(busy.stdout or "[]")]
    except Exception:
        statuses = []
    if any(s in ("in_progress", "queued", "waiting", "pending") for s in statuses):
        print("recorder already running or queued -- not restarting")
        return False
    r = gh("workflow", "run", "live_tick_recorder.yml", "--ref", "main")
    print("restart recorder:", r.returncode, (r.stderr or "").strip()[:200])
    return r.returncode == 0

def main():
    now = now_utc()
    problems, ok, ticks_age = evaluate(now)
    print("fresh:", ok); print("problems:", problems)
    restarted = False
    if ticks_age is None or ticks_age > RESTART_TICKS_AFTER_MIN:
        restarted = restart_recorder()
    gh("label", "create", LABEL, "--color", "d93f0b", "--description", "Live data stopped updating", "--force")
    found = gh("issue", "list", "--label", LABEL, "--state", "open", "--json", "number", "--limit", "1")
    try:
        existing = json.loads(found.stdout or "[]")
    except Exception:
        existing = []
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
    if problems:
        body = ("The live data has stopped updating:\n\n" + "\n".join(problems) +
                f"\n\nChecked at {now.isoformat(timespec='seconds')}." +
                ("\n\nThe watchdog has asked GitHub to restart the price recorder." if restarted else "") +
                "\n\nIf it does not recover, open **Actions** and run **Record Live Gold Ticks** and **Update Short-Term Ranges** by hand.")
        if existing:
            gh("issue", "comment", str(existing[0]["number"]), "--body", body)
        else:
            args = ["issue", "create", "--title", ISSUE_TITLE, "--label", LABEL, "--body", body]
            if owner: args += ["--assignee", owner]
            r = gh(*args); print("issue create:", r.returncode, (r.stderr or r.stdout).strip()[:200])
    elif existing:
        gh("issue", "close", str(existing[0]["number"]), "--comment", "Everything is updating again -- closing automatically.")

def selftest():
    import tempfile
    now = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    d = tempfile.mkdtemp()
    def w(f, hours_ago):
        json.dump({"updated_at": (now - timedelta(hours=hours_ago)).isoformat()}, open(os.path.join(d, f), "w"))
    for f, _, _ in CHECKS: w(f, 0.2)
    p, ok, t = evaluate(now, d); assert not p and len(ok) == len(CHECKS), (p, ok)
    w("live_gold_ticks.json", 2.5); w("short_range_latest.json", 1.0); w("gold_prediction_latest.json", 3.0)
    p, ok, t = evaluate(now, d)
    assert len(p) == 1 and "recorder" in p[0] and abs(t - 150) < 1, (p, t)
    os.remove(os.path.join(d, "direction_live_record.json"))
    p, ok, t = evaluate(now, d); assert len(p) == 2, p
    print("SELFTEST PASSED")

if __name__ == "__main__":
    selftest() if (len(sys.argv) > 1 and sys.argv[1] == "selftest") else main()
