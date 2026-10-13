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
    if run("git", "diff", "--staged", "--quiet").returncode == 0:
        return
    run("git", "commit", "-m", "Update live gold ticks [automated]")
    for i in range(5):
        if run("git", "push").returncode == 0:
            return
        run("git", "pull", "--rebase", "-X", "theirs", "origin", "main")
        time.sleep(2 + i)

def main():
    if not API_KEY:
        raise SystemExit("GOLDPRICEZ_API_KEY is not set.")
    d = load()
    end = time.time() + RUN_MINUTES * 60
    last_commit = time.time()
    while time.time() < end:
        t0 = time.time()
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
