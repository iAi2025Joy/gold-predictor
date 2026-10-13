"""
research_range_events.py -- RESEARCH ONLY (touches no prediction file).

Question: do our live 70% / 90% price ranges still hold at the busy moments (right after a sudden spike,
at 8:30 / 9:30 / 10:00 New York time data & open)? If they hold much less often there, how much wider
should they be?

Method: replays the EXACT production range method (short_range_updater.make_range) on ~60 days of real
5-minute bars, using only data available at each moment (no look-ahead), then checks whether the real price
landed inside the range H minutes later. Reported per moment type and horizon, with a binomial test vs the
target, plus the smallest widening factor that restores the target.

Run:  python research_range_events.py            (needs short_range_updater.py in the same folder)
      python research_range_events.py selftest
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import short_range_updater as sru
warnings.filterwarnings("ignore")

HORIZ = {"10min": 2, "20min": 4, "30min": 6, "60min": 12}
STRIDE = 3                      # replay every 3rd bar (keeps it fast; samples stay spread out)
FACTORS = [1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.8]
MIN_N = 40

def load_live():
    import yfinance as yf
    out = {}
    for name, t in sru.ASSETS.items():
        d = yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True)
        c = d["Close"]
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        c = c.astype(float).dropna()
        c.index = pd.to_datetime(c.index)
        c.index = c.index.tz_localize("UTC") if c.index.tz is None else c.index.tz_convert("UTC")
        out[name] = c[c > 0]
    return out

def synthetic(n=12000, burst=1.0, seed=5):
    """Constant-vol random walk; if burst>1, volatility is `burst` times larger for the 12 bars after every
    spike and after 08:30/09:30/10:00 ET -- a known weakness the test should detect."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-02", periods=n, freq="5min", tz="UTC")
    et = idx.tz_convert("America/New_York")
    sig = np.full(n, 0.0004)
    if burst > 1:
        for i in range(n):
            if (et[i].hour, et[i].minute) in ((8, 30), (9, 30), (10, 0)):
                sig[i + 1:i + 13] = 0.0004 * burst
    r = rng.normal(0, 1, n) * sig
    return pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)

def classify(price):
    """Label every bar: spike / data_open (8:30, 9:30, 10:00 NY) / normal. Uses only data up to that bar."""
    lp = np.log(price.values)
    r1 = np.diff(lp, prepend=np.nan)
    sd = pd.Series(r1).rolling(288, min_periods=100).std().shift(1).values
    z1 = np.abs(r1) / sd
    z3 = np.abs(lp - np.roll(lp, 3)) / (sd * np.sqrt(3))
    et = price.index.tz_convert("America/New_York")
    lab = []
    for i in range(len(price)):
        if (et[i].hour, et[i].minute) in ((8, 30), (9, 30), (10, 0)):
            lab.append("data_or_open")
        elif (np.isfinite(z1[i]) and z1[i] > 2.5) or (np.isfinite(z3[i]) and z3[i] > 2.5):
            lab.append("spike")
        else:
            lab.append("normal")
    return lab

def replay(asset, price):
    labels = classify(price)
    lp = np.log(price.values)
    start = sru.CAL_POINTS + sru.VOL_BARS + 20
    rows = []
    for t in range(start, len(price) - max(HORIZ.values()) - 1):
        if labels[t] == "normal" and t % STRIDE:
            continue                      # thin out only the ordinary moments; keep every special moment
        sub = price.iloc[:t + 1]
        for hname, H in HORIZ.items():
            m = sru.make_range(sub, H, asset)
            if not m:
                continue
            base = m["base_price"]
            actual = float(price.iloc[t + H])
            rows.append({"cat": labels[t], "h": hname, "base": base, "actual": actual,
                         "lo": m["range_low"], "hi": m["range_high"], "lo9": m["range_low90"], "hi9": m["range_high90"]})
    return pd.DataFrame(rows)

def widen(base, lo, hi, f):
    return base * np.exp(np.log(lo / base) * f), base * np.exp(np.log(hi / base) * f)

def summarise(asset, df):
    out = []
    for (cat, h), g in df.groupby(["cat", "h"]):
        if len(g) < MIN_N:
            continue
        hit70 = float(((g.actual >= g.lo) & (g.actual <= g.hi)).mean())
        hit90 = float(((g.actual >= g.lo9) & (g.actual <= g.hi9)).mean())
        k70 = int(((g.actual >= g.lo) & (g.actual <= g.hi)).sum())
        p = stats.binomtest(k70, len(g), 0.70).pvalue
        need70 = need90 = None
        for f in FACTORS:
            l, hh = widen(g.base, g.lo, g.hi, f)
            if need70 is None and ((g.actual >= l) & (g.actual <= hh)).mean() >= 0.70: need70 = f
            l9, h9 = widen(g.base, g.lo9, g.hi9, f)
            if need90 is None and ((g.actual >= l9) & (g.actual <= h9)).mean() >= 0.90: need90 = f
        out.append({"asset": asset, "moment": cat, "horizon": h, "n": int(len(g)),
                    "hit_rate_70band": round(hit70, 3), "hit_rate_90band": round(hit90, 3),
                    "p_vs_70": round(p, 4), "widen_factor_needed_70": need70, "widen_factor_needed_90": need90})
    return out

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        res = {}
        for name, b in (("constant_vol", 1.0), ("event_bursts_x2", 2.0)):
            res[name] = summarise("gold", replay("gold", synthetic(burst=b)))
            print(name)
            for x in res[name]:
                if x["horizon"] == "30min": print("  ", x["moment"], "n", x["n"], "70-band", x["hit_rate_70band"], "widen", x["widen_factor_needed_70"])
        norm = [x for x in res["constant_vol"] if x["moment"] == "normal" and x["horizon"] == "30min"][0]["hit_rate_70band"]
        ev = [x for x in res["event_bursts_x2"] if x["moment"] == "data_or_open" and x["horizon"] == "30min"]
        ok = 0.62 <= norm <= 0.78 and ev and ev[0]["hit_rate_70band"] < 0.60
        print("SELFTEST", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
    allrows = []
    for asset, price in load_live().items():
        print("replaying", asset, len(price), "bars ...", flush=True)
        allrows += summarise(asset, replay(asset, price))
    json.dump(allrows, open("research_range_events_report.json", "w"), indent=1, default=float)
    print(f"{'asset':5s} {'moment':13s} {'hor':6s} {'n':>5s} {'70%band':>8s} {'90%band':>8s} {'p':>7s} widen70 widen90")
    for x in sorted(allrows, key=lambda x: (x["asset"], x["horizon"], x["moment"])):
        print(f"{x['asset']:5s} {x['moment']:13s} {x['horizon']:6s} {x['n']:5d} {x['hit_rate_70band']:8.3f} {x['hit_rate_90band']:8.3f} {x['p_vs_70']:7.4f} x{x['widen_factor_needed_70']}  x{x['widen_factor_needed_90']}")

if __name__ == "__main__":
    main()
