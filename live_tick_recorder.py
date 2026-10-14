"""
live_tick_recorder.py -- records the LIVE gold spot price (GoldPriceZ) every 2 minutes for ~52 minutes,
committing to live_gold_ticks.json every ~10 minutes. Purpose: build our own fresh, un-delayed price history
(Yahoo futures bars are ~10 minutes old) so ranges and tests can start from "now".

Rate budget: 30 requests/hour here + 1/hour from the gold updater = 31 of the 60/hour allowed.
Needs env GOLDPRICEZ_API_KEY. Oil has no free live feed, so it is not recorded here.
"""
import os, json, time, subprocess, requests
from datetime import datetime, timezone, timedelta

API_KEY = os.environ.get("GOLDPRICEZ_API_KEY", "")
URL = "https://goldpricez.com/api/rates/currency/usd/measure/ounce"
OUT = "live_gold_ticks.json"
RUN_MINUTES = float(os.environ.get("RUN_MINUTES", "52"))
EVERY_SEC = int(os.environ.get("EVERY_SEC", "120"))
COMMIT_EVERY_SEC = 600
KEEP_DAYS = 30

def fetch():
    resp = requests.get(URL, headers={"X-API-KEY": API_KEY}, timeout=15)
    resp.raise_for_status()
    outer = resp.json()
    data = json.loads(outer) if isinstance(outer, str) else outer
    return float(data["ounce_price_usd"])

def load():
    try:
        return json.load(open(OUT))
    except Exception:
        return {"description": "Live gold spot ticks from GoldPriceZ (t = UTC ISO time, p = USD/oz).", "ticks": []}

def save(d):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).isoformat()
    d["ticks"] = [t for t in d["ticks"] if t["t"] >= cutoff]
    d["updated_at"] = datetime.now(timezone.utc).isoformat()
    json.dump(d, open(OUT, "w"), separators=(",", ":"))

def git_push():
    def run(*a):
        return subprocess.run(a, capture_output=True, text=True)
    run("git", "config", "user.name", "gold-predictor-bot")
    run("git", "config", "user.email", "actions@github.com")
    run("git", "add", OUT)
    for f in ("direction_track_record.json", "direction_live_record.json", "short_range_latest.json", "short_range_track_record.json", "event_calendar.json", "calibration_state.json"):
        if os.path.exists(f):
            run("git", "add", f)
    if run("git", "diff", "--staged", "--quiet").returncode == 0:
        return
    run("git", "commit", "-m", "Update live gold ticks [automated]")
    for i in range(5):
        if run("git", "push").returncode == 0:
            return
        run("git", "pull", "--rebase", "-X", "theirs", "origin", "main")
        time.sleep(2 + i)

def direction_cycle():
    """Every ~10 minutes: score due up/down calls and make a new one (see direction_record.py). Never breaks tick recording."""
    try:
        import direction_record
        direction_record.run_once()
    except Exception as e:
        print("direction cycle failed:", e)

def ranges_cycle():
    """Every ~15 minutes: refresh short_range_latest.json (GitHub often skips the separate 15-minute schedule)."""
    try:
        r = subprocess.run(["python", "short_range_updater.py"], capture_output=True, text=True, timeout=300)
        print("ranges refresh exit", r.returncode, (r.stderr or "")[-200:])
    except Exception as e:
        print("ranges cycle failed:", e)

def calendar_cycle():
    """Every ~6 hours: refresh event_calendar.json (a second path besides its own workflow)."""
    try:
        r = subprocess.run(["python", "event_calendar_fetcher.py"], capture_output=True, text=True, timeout=120)
        print("calendar refresh exit", r.returncode)
    except Exception as e:
        print("calendar cycle failed:", e)

def main():
    if not API_KEY:
        raise SystemExit("GOLDPRICEZ_API_KEY is not set.")
    d = load()
    end = time.time() + RUN_MINUTES * 60
    last_commit = time.time()
    next_dir = 0.0
    next_rng = 0.0
    next_cal = 0.0
    while time.time() < end:
        t0 = time.time()
        if t0 >= next_dir:
            direction_cycle()
            next_dir = t0 + 600
        if t0 >= next_rng:
            ranges_cycle()
            next_rng = t0 + 900
        if t0 >= next_cal:
            calendar_cycle()
            next_cal = t0 + 6 * 3600
        try:
            p = fetch()
            d["ticks"].append({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "p": round(p, 2)})
            print("tick", d["ticks"][-1])
        except Exception as e:
            print("fetch failed:", e)
        if time.time() - last_commit >= COMMIT_EVERY_SEC:
            save(d); git_push(); last_commit = time.time()
        time.sleep(max(1, EVERY_SEC - (time.time() - t0)))
    save(d); git_push()

if __name__ == "__main__":
    main()
