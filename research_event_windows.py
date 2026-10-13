"""
research_event_windows.py -- RESEARCH ONLY (touches no prediction file).

Question: is there ANY special moment where the next 10-60 minutes of gold/oil direction is
predictable better than a coin flip -- right after a sudden spike, or around scheduled data/
announcement times? And do those moments need WIDER price ranges?

Strict rules: only past data defines a trigger; triggers are thinned so outcome windows never overlap;
every test is Bonferroni-corrected across ALL tests run; a result counts only if it is significant
overall AND has the same sign in both halves of the data. A built-in positive control injects a known
edge and checks the test can find it (otherwise "found nothing" would mean nothing).

Data: Yahoo 5-minute futures bars (GC=F, CL=F), last ~60 days (Yahoo's limit).
Output: research_event_windows_report.json + printed summary.   Selftest: python research_event_windows.py selftest
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ASSETS = {"gold": "GC=F", "oil": "CL=F"}
HORIZONS = {"10min": 2, "20min": 4, "30min": 6, "60min": 12}   # in 5-minute bars
ALPHA = 0.05
MIN_N = 30

def load_live():
    import yfinance as yf
    out = {}
    for name, t in ASSETS.items():
        df = yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True)
        c = df["Close"]
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        c = c.astype(float).dropna()
        c.index = pd.to_datetime(c.index)
        if c.index.tz is None: c.index = c.index.tz_localize("UTC")
        out[name] = c.tz_convert("America/New_York")
    return out

def synthetic(n=16000, spike_edge=0.0, seed=3):
    """Random-walk prices; optionally after a big spike the next hour continues the spike's direction."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-02", periods=n, freq="5min", tz="America/New_York")
    r = rng.normal(0, 0.0004, n)
    if spike_edge:
        sd = pd.Series(r).rolling(288, min_periods=50).std().shift(1).values
        i = 300
        while i < n - 14:
            if abs(r[i]) > 2.5 * sd[i]:
                drift = np.sign(r[i]) * spike_edge * 0.0004
                r[i + 1:i + 13] += drift
                i += 13
            i += 1
    return pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)

def build(c):
    r = np.log(c).diff()
    sd = r.rolling(288, min_periods=100).std().shift(1)
    return r, sd

def thin(idx_positions, gap=13):
    out, last = [], -10**9
    for p in idx_positions:
        if p - last >= gap:
            out.append(p); last = p
    return out

def triggers(c, r, sd):
    """name -> list of integer bar positions where the condition fired (using only data up to that bar)."""
    pos = np.arange(len(c))
    z = (r / sd).values
    et = c.index
    T = {}
    T["spike_1bar_z>2.5"] = thin([p for p in pos if np.isfinite(z[p]) and abs(z[p]) > 2.5])
    z3 = ((np.log(c) - np.log(c).shift(3)) / (sd * np.sqrt(3))).values
    T["spike_15min_z>2.5"] = thin([p for p in pos if np.isfinite(z3[p]) and abs(z3[p]) > 2.5])
    def at(hh, mm):
        return thin([p for p in pos if et[p].hour == hh and et[p].minute == mm and np.isfinite(z[p])])
    T["after_0830_data"] = at(8, 30)
    T["after_1000_data"] = at(10, 0)
    T["after_1400_fed_slot"] = at(14, 0)
    T["ny_open_0930"] = at(9, 30)
    return T

def test_condition(c, r, name, bars, h):
    """Continuation test: did the next-h move keep the direction of the trigger bar's move? (rate<0.5 = reversal)"""
    logc = np.log(c).values
    hits, ts = [], []
    for p in bars:
        if p + h >= len(c) or not np.isfinite(r.values[p]) or r.values[p] == 0:
            continue
        fut = logc[p + h] - logc[p]
        if fut == 0:
            continue
        hits.append(int(np.sign(fut) == np.sign(r.values[p]))); ts.append(c.index[p])
    n = len(hits)
    if n < MIN_N:
        return {"n": n, "note": "too few events"}
    k = sum(hits)
    p = stats.binomtest(k, n, 0.5).pvalue
    half = n // 2
    r1, r2 = np.mean(hits[:half]), np.mean(hits[half:])
    return {"n": n, "continuation_rate": round(k / n, 3), "p_value": p,
            "first_half_rate": round(float(r1), 3), "second_half_rate": round(float(r2), 3),
            "same_side_both_halves": bool((r1 - 0.5) * (r2 - 0.5) > 0 and (r1 - 0.5) * (k / n - 0.5) > 0)}

def vol_ratio(c, r, bars, h):
    """Typical |move| over the next h bars after these moments vs all moments (>1 means ranges should be wider)."""
    logc = np.log(c).values
    allmv = np.abs(logc[h:] - logc[:-h]); allmv = allmv[np.isfinite(allmv)]
    mv = [abs(logc[p + h] - logc[p]) for p in bars if p + h < len(c)]
    if len(mv) < MIN_N:
        return None
    return round(float(np.median(mv) / np.median(allmv)), 2)

def run(series_by_asset, label):
    results, ntests = [], 0
    for asset, c in series_by_asset.items():
        r, sd = build(c)
        T = triggers(c, r, sd)
        for tname, bars in T.items():
            for hname, h in HORIZONS.items():
                res = test_condition(c, r, tname, bars, h)
                res.update({"asset": asset, "condition": tname, "horizon": hname,
                            "range_width_ratio": vol_ratio(c, r, bars, h)})
                if "p_value" in res: ntests += 1
                results.append(res)
    for res in results:
        if "p_value" in res:
            res["p_bonferroni"] = min(1.0, res["p_value"] * ntests)
            res["real_edge"] = bool(res["p_bonferroni"] < ALPHA and res["same_side_both_halves"])
    found = [x for x in results if x.get("real_edge")]
    return {"label": label, "tests_run": ntests, "edges_found": len(found), "found": found, "all": results}

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        null = run({"synthetic_no_edge": synthetic()}, "selftest: random walk (expect 0 edges)")
        ctl = run({"synthetic_with_edge": synthetic(spike_edge=0.35)}, "selftest: injected spike-continuation edge (expect edges)")
        print(null["label"], "->", null["edges_found"], "edges of", null["tests_run"], "tests")
        print(ctl["label"], "->", ctl["edges_found"], "edges of", ctl["tests_run"], "tests")
        ok = null["edges_found"] == 0 and ctl["edges_found"] > 0
        print("SELFTEST", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    live = run(load_live(), "live Yahoo 5-minute bars, ~60 days")
    ctl = run({"positive_control": synthetic(spike_edge=0.35)}, "positive control (known injected edge)")
    report = {"live": {k: v for k, v in live.items() if k != "all"}, "control_edges_found": ctl["edges_found"],
              "all_tests": live["all"]}
    json.dump(report, open("research_event_windows_report.json", "w"), indent=1, default=float)
    print("Positive control found", ctl["edges_found"], "edges (must be >0 for the test to mean anything).")
    print("LIVE: tests run =", live["tests_run"], "| real edges found =", live["edges_found"])
    for x in live["found"]:
        print("  EDGE:", x["asset"], x["condition"], x["horizon"], "rate", x["continuation_rate"], "n", x["n"], "p_bonf", round(x["p_bonferroni"], 4))
    print("\nRange-width ratio (typical move after the moment vs normal; >1.2 means widen ranges):")
    for x in live["all"]:
        if x.get("range_width_ratio") and x["horizon"] == "30min":
            print(f"  {x['asset']:5s} {x['condition']:22s} x{x['range_width_ratio']}  (n={x['n']})")
    print("\nVerdict:", "EDGE FOUND -- see list above" if live["edges_found"] else "NO reliable edge at any special moment.")

if __name__ == "__main__":
    main()
