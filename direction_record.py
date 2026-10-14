"""
direction_record.py -- an honest, self-scoring LIVE direction track record.

Every 10 minutes it records an UP or DOWN call for gold and oil (the same call GARNET shows: Up if the share
of similar recent 10-minute windows that ended higher is >= 50%, else Down). Ten minutes after the call's
start time it looks up the real price move and marks the call right or wrong. It keeps only the newest 1000
SCORED calls per asset; accuracy = right calls / scored calls.

Fairness rules (so the number cannot flatter itself):
  * A call uses only bars that existed when it was made (Yahoo bars are ~10 min late; the call is about the
    move over the 10 minutes starting at the call's own time, which is still in the future of its data).
  * Outcome = close of the bar 10 minutes later vs the close of the bar at the call time, looked up AFTER the fact.
  * Unchanged prices ("flat") and missing bars (market closed) are not counted either way.
  * Nothing is ever edited after scoring; old calls just roll off after 1000.
Files written: direction_track_record.json (calls) and direction_live_record.json (summary read by GARNET).
Run once:  python direction_record.py once      Self-test:  python direction_record.py selftest
"""
import sys, json, os, warnings
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

RECORD_FILE = "direction_track_record.json"
SUMMARY_FILE = "direction_live_record.json"
KEEP = 1000
HOR_MIN = 10
EXPIRE_HOURS = 4
MAX_BAR_AGE_MIN = 60          # no new call if the newest bar is older than this (market closed)

def _floor5(ts):
    return ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0)

def make_call(asset, price, now, make_range):
    """price: completed 5-minute bars (UTC index, close prices). Returns a pending record or None."""
    if price is None or len(price) < 700:
        return None
    if now - price.index[-1].to_pydatetime() > timedelta(minutes=MAX_BAR_AGE_MIN):
        return None
    m = make_range(price, HOR_MIN // 5, asset)
    if not m or m.get("chance_higher") is None:
        return None
    t_end = _floor5(now)
    base_bar = t_end - timedelta(minutes=5)          # bar that ends at the call time
    target_bar = t_end + timedelta(minutes=5)        # bar that ends 10 minutes later
    return {"id": f"{asset}-{base_bar.isoformat()}", "asset": asset, "made_at": now.isoformat(timespec="seconds"),
            "base_bar": base_bar.isoformat(), "target_bar": target_bar.isoformat(),
            "predicted": "up" if m["chance_higher"] >= 0.5 else "down",
            "chance_up": m["chance_higher"], "status": "pending"}

def score(records, prices, now):
    for r in records:
        if r["status"] != "pending":
            continue
        p = prices.get(r["asset"])
        b, t = pd.Timestamp(r["base_bar"]), pd.Timestamp(r["target_bar"])
        if p is not None and b in p.index and t in p.index:
            pb, pt = float(p.loc[b]), float(p.loc[t])
            r.update({"base_price": round(pb, 4), "target_price": round(pt, 4)})
            if pt == pb:
                r["status"] = "flat"
            else:
                r["outcome"] = "up" if pt > pb else "down"
                r["correct"] = r["outcome"] == r["predicted"]
                r["status"] = "scored"
                r["scored_at"] = now.isoformat(timespec="seconds")
        elif now - t.to_pydatetime() > timedelta(hours=EXPIRE_HOURS):
            r["status"] = "expired"
    return records

def prune(records):
    out = []
    for asset in {r["asset"] for r in records}:
        rs = [r for r in records if r["asset"] == asset]
        scored = sorted([r for r in rs if r["status"] == "scored"], key=lambda r: r["target_bar"])[-KEEP:]
        pend = [r for r in rs if r["status"] == "pending"]
        out += scored + pend
    return sorted(out, key=lambda r: (r["asset"], r["target_bar"]))

def summarise(records, now):
    res = {"updated_at": now.isoformat(timespec="seconds"), "window": f"last {KEEP} scored 10-minute calls",
           "how_to_read": "Each call is Up or Down for the next 10 minutes, scored against the real price 10 minutes later. "
                          "50% is a coin flip. Unchanged prices and market-closed periods are not counted.",
           "assets": {}}
    for asset in sorted({r["asset"] for r in records}):
        sc = [r for r in records if r["asset"] == asset and r["status"] == "scored"]
        n = len(sc); k = sum(1 for r in sc if r["correct"])
        a = {"n_scored": n, "n_correct": k, "pending": sum(1 for r in records if r["asset"] == asset and r["status"] == "pending")}
        if n:
            ci = stats.binomtest(k, n, 0.5).proportion_ci(confidence_level=0.95, method="wilson")
            a.update({"accuracy": round(k / n, 4), "ci95_low": round(ci.low, 4), "ci95_high": round(ci.high, 4),
                      "p_vs_coin_flip": round(stats.binomtest(k, n, 0.5).pvalue, 4),
                      "share_of_calls_up": round(sum(1 for r in sc if r["predicted"] == "up") / n, 3),
                      "share_of_outcomes_up": round(sum(1 for r in sc if r["outcome"] == "up") / n, 3),
                      "first_call": min(r["made_at"] for r in sc), "last_call": max(r["made_at"] for r in sc)})
            base = max(a["share_of_outcomes_up"], 1 - a["share_of_outcomes_up"])
            a["baseline_majority_accuracy"] = round(base, 3)
            a["statement"] = (f"Live direction record: {a['accuracy']*100:.0f}% of the last {n} ten-minute up/down calls were right "
                              f"(a coin flip is 50%; simply always guessing the more common direction would have scored {base*100:.0f}%; "
                              f"with this many calls, normal luck is about +/-{(a['ci95_high']-a['ci95_low'])*50:.0f} points).")
        res["assets"][asset] = a
    return res

def load_records():
    try:
        return json.load(open(RECORD_FILE))["calls"]
    except Exception:
        return []

def save_all(records, now):
    json.dump({"note": "Newest 1000 scored calls per asset plus pending ones.", "calls": records}, open(RECORD_FILE, "w"), separators=(",", ":"))
    json.dump(summarise(records, now), open(SUMMARY_FILE, "w"), indent=1)

def cycle(prices, now, make_range, records=None):
    """One 10-minute cycle: score what is due, add a new call per asset, prune, return records."""
    records = load_records() if records is None else records
    records = score(records, prices, now)
    have = {r["id"] for r in records}
    for asset, p in prices.items():
        c = make_call(asset, p, now, make_range)
        if c and c["id"] not in have:
            records.append(c)
    return prune(records)

def run_once():
    import short_range_updater as sru
    now = datetime.now(timezone.utc)
    prices = {}
    for asset, tk in sru.ASSETS.items():
        try:
            prices[asset] = sru.fetch_bars(tk, now)
        except Exception as e:
            print("bars failed", asset, e)
    recs = cycle(prices, now, sru.make_range)
    save_all(recs, now)
    print(json.dumps(summarise(recs, now)["assets"], indent=1))

# ---------------------------------------------------------------- selftest
def _sim(drift=0.0, days=6, seed=11):
    rng = np.random.default_rng(seed)
    n = days * 288 + 900
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    r = rng.normal(drift, 0.0004, n)
    return pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)

def selftest():
    import short_range_updater as sru
    out = {}
    for name, drift in (("no_edge", 0.0), ("with_drift", 0.00012)):
        full = _sim(drift)
        recs, cur = [], full.index[800].to_pydatetime() + timedelta(minutes=3)
        end = full.index[-1].to_pydatetime() - timedelta(minutes=30)
        while cur < end:
            delayed = full[full.index <= pd.Timestamp(cur) - timedelta(minutes=10)]
            avail = full[full.index <= pd.Timestamp(cur)]          # bars that exist by 'now' (for scoring)
            recs = score(recs, {"gold": avail}, cur)
            c = make_call("gold", delayed, cur, sru.make_range)
            if c and c["id"] not in {r["id"] for r in recs}:
                recs.append(c)
            recs = prune(recs)
            cur += timedelta(minutes=10)
        s = summarise(recs, cur)["assets"]["gold"]
        out[name] = s
        print(name, "scored", s["n_scored"], "accuracy", s["accuracy"], "ci", s["ci95_low"], s["ci95_high"])
    ok = (out["no_edge"]["n_scored"] <= KEEP and 0.44 <= out["no_edge"]["accuracy"] <= 0.56 and out["with_drift"]["accuracy"] > 0.56)
    print("SELFTEST", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest": selftest()
    else: run_once()
