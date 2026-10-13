"""
research_range_sharpness.py -- RESEARCH ONLY. Touches no prediction file.

Goal: make the 70% ranges TIGHTER while they still contain the real price ~70% of the time.
A tighter range at the same coverage is a genuinely better forecast (more informative).

Compares volatility estimators for 10/20/30/60-minute ranges, gold (GC=F) and oil (CL=F), on ~60 days of real
5-minute bars, walk-forward (only past data at each point):
  base      : std of last 36 bars (what is live today)
  long      : std of last 144 bars (12 hours)
  ewma      : exponentially weighted std (lambda 0.97 per bar)
  tod       : base, de-seasonalised then re-seasonalised for the UPCOMING time-of-day (vol is higher at the US open etc.)
  ewma_tod  : ewma with the same time-of-day correction
For each: coverage (target 70%), mean width in basis points, and interval score (width + penalty for misses; LOWER is better).
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ASSETS = {"gold": "GC=F", "oil": "CL=F"}
HORIZONS_MIN = [10, 20, 30, 60]
COVERAGE, ALPHA = 0.70, 0.30
CAL_POINTS, MIN_Z = 600, 150
VARIANTS = ["base", "long", "ewma", "tod", "ewma_tod"]


def load_live(ticker):
    import yfinance as yf
    d = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)
    c = d["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    c = c.astype(float).dropna(); c.index = pd.to_datetime(c.index)
    return c[c > 0]


def load_selftest(seed=4, n=15000):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    seas = 1 + 0.8 * np.exp(-((idx.hour + idx.minute / 60 - 14) ** 2) / 4)       # vol spike ~14:00 UTC
    return pd.Series(2000 * np.exp(np.cumsum(rng.standard_t(5, n) * 0.0005 * seas)), index=idx)


def tod_factor(price):
    """Causal time-of-day vol factor per bar: past RMS return in that 30-min bucket / past overall RMS."""
    r = np.diff(np.log(price.values), prepend=np.nan)
    b = (price.index.hour * 2 + price.index.minute // 30).values
    sums, cnts = np.zeros(48), np.zeros(48); tot_s, tot_c = 0.0, 0
    f = np.ones(len(r))
    for i in range(len(r)):
        if tot_c > 500 and cnts[b[i]] > 20:
            f[i] = np.sqrt(sums[b[i]] / cnts[b[i]]) / np.sqrt(tot_s / tot_c)
        if np.isfinite(r[i]):                   # update AFTER use: strictly past information
            sums[b[i]] += r[i] ** 2; cnts[b[i]] += 1; tot_s += r[i] ** 2; tot_c += 1
    return np.clip(f, 0.3, 3.0), b


def vol_series(price, variant, H):
    r = pd.Series(np.diff(np.log(price.values), prepend=np.nan))
    if variant in ("base", "tod"): v = r.rolling(36).std().values
    elif variant == "long": v = r.rolling(144).std().values
    else: v = np.sqrt((r ** 2).ewm(alpha=0.03, adjust=False).mean().shift(0).values)
    if variant in ("tod", "ewma_tod"):
        f, b = tod_factor(price); n = len(r)
        fut = np.ones(n)
        for i in range(n):                       # average relative variance over the H upcoming bars (known clock times)
            idxs = [(b[i] + (k * 5 + price.index[i].minute % 30) // 30) % 48 for k in range(1, H + 1)]
            # use the factor table at the same clock buckets taken from the most recent day's values
            fut[i] = np.sqrt(np.mean([f[max(i - 288 + k, 0)] ** 2 for k in range(1, H + 1)])) if i >= 288 else 1.0
        v = v / f * fut
    return v


def evaluate(price, H, variant):
    lp = np.log(price.values); n = len(lp)
    vol = vol_series(price, variant, H)
    rH = np.full(n, np.nan); rH[:n - H] = lp[H:] - lp[:-H]
    rows = []
    for t in range(CAL_POINTS + 150 + H, n - H, H):
        js = np.arange(t - CAL_POINTS - H, t - H + 1)
        z = rH[js] / (vol[js] * np.sqrt(H)); z = z[np.isfinite(z)]
        if len(z) < MIN_Z or not np.isfinite(vol[t]) or vol[t] <= 0: continue
        lo, hi = np.quantile(z, [(1 - COVERAGE) / 2, 1 - (1 - COVERAGE) / 2])
        s = vol[t] * np.sqrt(H); L, U, y = lo * s, hi * s, lp[t + H] - lp[t]
        width = (U - L) * 1e4
        pen = (2 / ALPHA) * max(L - y, 0) * 1e4 + (2 / ALPHA) * max(y - U, 0) * 1e4
        rows.append((L <= y <= U, width, width + pen))
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2]


def main():
    selftest = "--selftest" in sys.argv
    report = {"results": {}}
    for name, tk in ASSETS.items():
        price = load_selftest() if selftest else load_live(tk)
        print(f"\n{name} ({tk}): {len(price)} bars")
        for m in HORIZONS_MIN:
            H = m // 5; res = {}
            for v in VARIANTS:
                hit, w, sc = evaluate(price, H, v)
                res[v] = {"n": int(len(hit)), "coverage": float(hit.mean()), "mean_width_bps": float(w.mean()), "interval_score": float(sc.mean())}
            base = res["base"]["interval_score"]
            for v in VARIANTS:
                res[v]["score_vs_base_pct"] = round((res[v]["interval_score"] / base - 1) * 100, 1)
                res[v]["coverage_ok"] = bool(abs(res[v]["coverage"] - COVERAGE) <= 0.05)
            ok = [v for v in VARIANTS if res[v]["coverage_ok"]]
            best = min(ok, key=lambda v: res[v]["interval_score"]) if ok else "base"
            report["results"][f"{name}|{m}min"] = {"variants": res, "best": best}
            print(f"  {m:3d} min  best={best}")
            for v in VARIANTS:
                r = res[v]
                print(f"     {v:9s} cov={r['coverage']:.3f} width={r['mean_width_bps']:6.1f}bps score={r['interval_score']:7.1f} ({r['score_vs_base_pct']:+.1f}% vs base){' <--' if v == best else ''}")
    json.dump(report, open("research_range_sharpness_report.json", "w"), indent=2)
    print("\nDone. Lower interval score = tighter range at the same ~70% coverage.")

if __name__ == "__main__":
    main()
