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
SELF-LEARNING (safe version): besides the live rule, six other simple rules (momentum, reversal, longer trend, recent-outcome
follow...) make a "shadow" call each cycle, all scored on the same real outcomes. A rule is only called VALIDATED (and only then
becomes the champion) if over >= 300 scored calls it beats a coin flip AND the "always guess the commoner direction" baseline,
with a Bonferroni-corrected p < 0.05 for having tried several rules, and also is above 50% in the most recent third of its calls.
Otherwise the system says plainly that no rule has beaten a coin flip. This is how it learns without fooling itself.
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

RULES = ["analog", "momentum_15m", "reversal_15m", "momentum_60m", "reversal_60m", "follow_recent_outcomes",
         "news_follow", "news_fade"]
NEWS_FILE = "market_headlines.json"
NEWS_WINDOW_H = 6          # headlines newer than this count
NEWS_DEADBAND = 0.15       # |average sentiment| below this = no opinion


def news_signal(asset, now):
    """Average VADER sentiment of this asset's headlines from the last 6 h (from market_headlines.json), or None.
    Returned as {score, n}. Used ONLY as two shadow rules that are scored live; it moves no price or range."""
    try:
        d = json.load(open(NEWS_FILE))
        sc = []
        for h in d.get(asset, []):
            t = datetime.fromisoformat(h["published"]).astimezone(timezone.utc)
            if timedelta(0) <= now - t <= timedelta(hours=NEWS_WINDOW_H) and h.get("sentiment") is not None:
                sc.append(float(h["sentiment"]))
        return {"score": round(sum(sc) / len(sc), 4), "n": len(sc)} if sc else None
    except Exception:
        return None
MIN_VALIDATE_N = 300


def shadow_calls(price, chance_higher, recent_outcomes, news=None):
    """Every candidate rule's UP/DOWN call from data available now (None when a rule has no opinion)."""
    c = price.values.astype(float)
    def sgn(x):
        return None if x == 0 or not np.isfinite(x) else ("up" if x > 0 else "down")
    flip = {"up": "down", "down": "up", None: None}
    m15 = sgn(c[-1] - c[-4]); m60 = sgn(c[-1] - c[-13])
    rec = None
    if recent_outcomes and len(recent_outcomes) >= 20:
        ups = sum(1 for o in recent_outcomes[-30:] if o == "up")
        rec = "up" if ups * 2 > len(recent_outcomes[-30:]) else ("down" if ups * 2 < len(recent_outcomes[-30:]) else None)
    nf = None
    if news and news.get("n", 0) >= 1 and abs(news["score"]) >= NEWS_DEADBAND:
        nf = "up" if news["score"] > 0 else "down"
    return {"analog": "up" if chance_higher >= 0.5 else "down", "momentum_15m": m15, "reversal_15m": flip[m15],
            "momentum_60m": m60, "reversal_60m": flip[m60], "follow_recent_outcomes": rec,
            "news_follow": nf, "news_fade": flip[nf]}


def make_call(asset, price, now, make_range, recent_outcomes=None, news=None):
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
            "chance_up": m["chance_higher"], "status": "pending",
            "news": news, "shadow": shadow_calls(price, m["chance_higher"], recent_outcomes, news)}

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
        a["learning"] = rule_leaderboard(sc, a.get("baseline_majority_accuracy", 0.5) if n else 0.5)
        if n and a["learning"]["verdict"]:
            a["statement"] += " " + a["learning"]["verdict"]
        res["assets"][asset] = a
    return res

def rule_leaderboard(scored, baseline):
    """Scores every shadow rule on the same outcomes and decides honestly whether any is validated."""
    rows, n_rules = [], len(RULES)
    ordered = sorted(scored, key=lambda r: r["target_bar"])
    for rule in RULES:
        calls = []
        for r in ordered:
            pred = r["shadow"].get(rule) if r.get("shadow") else (r["predicted"] if rule == "analog" else None)
            if pred is not None:
                calls.append(pred == r["outcome"])
        n = len(calls); k = sum(calls)
        if n == 0:
            continue
        acc = k / n
        p0 = max(0.5, baseline)
        p_coin = stats.binomtest(k, n, 0.5).pvalue
        p_base = stats.binomtest(k, n, p0, alternative="greater").pvalue
        third = calls[-max(1, n // 3):]
        recent_acc = sum(third) / len(third)
        validated = (n >= MIN_VALIDATE_N and acc > p0 and min(1.0, p_base * n_rules) < 0.05 and recent_acc > 0.5)
        rows.append({"rule": rule, "n": n, "accuracy": round(acc, 4), "recent_third_accuracy": round(recent_acc, 4),
                     "p_vs_coin_flip": round(p_coin, 4), "p_vs_baseline_corrected": round(min(1.0, p_base * n_rules), 4),
                     "validated": bool(validated)})
    rows.sort(key=lambda r: -r["accuracy"])
    val = [r for r in rows if r["validated"]]
    champ = val[0]["rule"] if val else None
    if champ:
        verdict = f"Rule check: '{champ}' has beaten both a coin flip and the always-guess-the-commoner baseline with statistical confidence over {val[0]['n']} calls."
    elif rows and max(r["n"] for r in rows) >= MIN_VALIDATE_N:
        verdict = f"Rule check: {len(rows)} simple rules were tested side by side and none has beaten a coin flip with statistical confidence."
    else:
        verdict = f"Rule check: {len(rows)} simple rules are being tested side by side; at least {MIN_VALIDATE_N} calls are needed before any can be judged."
    return {"champion": champ, "rules_tested": len(rows), "leaderboard": rows, "verdict": verdict}


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
        recent = [r["outcome"] for r in sorted((x for x in records if x["asset"] == asset and x["status"] == "scored"),
                                               key=lambda x: x["target_bar"])]
        c = make_call(asset, p, now, make_range, recent_outcomes=recent, news=news_signal(asset, now))
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
def _sim(drift=0.0, days=6, seed=11, rho=0.0):
    rng = np.random.default_rng(seed)
    n = days * 288 + 900
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    e = rng.normal(drift, 0.0004, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = rho * r[i - 1] + e[i]          # rho > 0 = real short-term momentum (a genuine, learnable edge)
    return pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)

def selftest():
    import short_range_updater as sru
    out = {}
    for name, (drift, rho) in (("no_edge", (0.0, 0.0)), ("with_drift", (0.0, 0.8))):
        full = _sim(drift, rho=rho)
        recs, cur = [], full.index[800].to_pydatetime() + timedelta(minutes=3)
        end = full.index[-1].to_pydatetime() - timedelta(minutes=30)
        while cur < end:
            delayed = full[full.index <= pd.Timestamp(cur) - timedelta(minutes=10)]
            avail = full[full.index <= pd.Timestamp(cur)]          # bars that exist by 'now' (for scoring)
            recs = score(recs, {"gold": avail}, cur)
            rec_out = [r["outcome"] for r in sorted((x for x in recs if x["status"] == "scored"), key=lambda x: x["target_bar"])]
            c = make_call("gold", delayed, cur, sru.make_range, recent_outcomes=rec_out)
            if c and c["id"] not in {r["id"] for r in recs}:
                recs.append(c)
            recs = prune(recs)
            cur += timedelta(minutes=10)
        s = summarise(recs, cur)["assets"]["gold"]
        out[name] = s
        print(name, "scored", s["n_scored"], "accuracy", s["accuracy"], "ci", s["ci95_low"], s["ci95_high"], "| champion:", s["learning"]["champion"])
        for r in s["learning"]["leaderboard"]:
            print("    ", r["rule"], r["n"], r["accuracy"], "validated" if r["validated"] else "")
    ok = (out["no_edge"]["n_scored"] <= KEEP and 0.44 <= out["no_edge"]["accuracy"] <= 0.56 
          and out["no_edge"]["learning"]["champion"] is None and (out["with_drift"]["learning"]["champion"] or "").startswith("momentum"))
    print("SELFTEST", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest": selftest()
    else: run_once()
