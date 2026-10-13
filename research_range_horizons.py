"""
research_range_horizons.py -- RESEARCH ONLY. Touches no prediction file.

Question: do calibrated 70% price ranges for 10 / 20 / 30 / 60 minutes really
contain the future price ~70% of the time, for gold and oil?

Method (same as the live updater's range): volatility per sqrt-time from recent
5-minute bars x the empirical 15th/85th percentile of past standardized
H-minute moves. Walk-forward: only past data used at each point.
Data: Yahoo 5-minute bars (~60 days): GC=F (gold), CL=F (WTI oil).
Also reports coverage by trading session, because intraday volatility is not constant.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ASSETS = {"gold": "GC=F", "oil": "CL=F"}
HORIZONS_MIN = [10, 20, 30, 60]
COVERAGE = 0.70
VOL_BARS = 36        # 3 hours of 5-min bars
CAL_POINTS = 600     # past standardized moves used for the z-quantiles

def load_live(ticker):
    import yfinance as yf
    d = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)
    c = d["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    c = c.astype(float).dropna(); c.index = pd.to_datetime(c.index)
    return c[c > 0]

def load_selftest(seed=2, n=15000):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    vol = 0.0006 * (1 + 0.5 * np.sin(np.arange(n) / 500))
    return pd.Series(2000 * np.exp(np.cumsum(rng.standard_t(5, n) * vol)), index=idx)

def session(ts):
    h = ts.hour
    return "asia_00-07UTC" if h < 7 else ("europe_07-13UTC" if h < 13 else "us_13-24UTC")

def evaluate(price, H):
    lp = np.log(price.values); r1 = np.diff(lp, prepend=np.nan)
    vol = pd.Series(r1).rolling(VOL_BARS).std().values
    rH = lp[H:] - lp[:-H]                      # move over H bars starting at i -> index i
    n = len(lp); hits, sess = [], []
    for t in range(CAL_POINTS + VOL_BARS + H, n - H, H):
        # past standardized H-bar moves fully known at time t
        js = np.arange(t - CAL_POINTS - H, t - H + 1)
        v = vol[js]
        z = rH[js] / (v * np.sqrt(H))
        z = z[np.isfinite(z)]
        if len(z) < 200 or not np.isfinite(vol[t]) or vol[t] <= 0:
            continue
        lo, hi = np.quantile(z, [(1 - COVERAGE) / 2, 1 - (1 - COVERAGE) / 2])
        move = lp[t + H] - lp[t]
        hits.append(lo * vol[t] * np.sqrt(H) <= move <= hi * vol[t] * np.sqrt(H))
        sess.append(session(price.index[t]))
    return np.array(hits), np.array(sess)

def main():
    selftest = "--selftest" in sys.argv
    report = {"coverage_target": COVERAGE, "results": {}}
    for name, tk in ASSETS.items():
        price = load_selftest() if selftest else load_live(tk)
        print(f"\n{name} ({tk}): {len(price)} bars {price.index.min()} -> {price.index.max()}")
        for m in HORIZONS_MIN:
            H = m // 5
            hits, sess = evaluate(price, H)
            if len(hits) < 100:
                print(f"  {m:3d} min: too little data"); continue
            cov = hits.mean(); k = int(hits.sum()); n = len(hits)
            p = stats.binomtest(k, n, COVERAGE).pvalue
            by = {s: round(float(hits[sess == s].mean()), 3) for s in np.unique(sess) if (sess == s).sum() >= 50}
            ok = abs(cov - COVERAGE) <= 0.05 and p > 0.01
            report["results"][f"{name}|{m}min"] = {"n": n, "coverage": float(cov), "p_vs_70": float(p),
                                                   "by_session": by, "calibrated": bool(ok)}
            print(f"  {m:3d} min: coverage={cov:.3f} n={n} p={p:.3f} {'OK' if ok else 'OFF-TARGET'}  sessions={by}")
    good = [k for k, v in report["results"].items() if v["calibrated"]]
    bad = [k for k, v in report["results"].items() if not v["calibrated"]]
    report["verdict"] = f"Calibrated (70% +/-5pts): {good or 'none'}. Off-target: {bad or 'none'}."
    print("\nVERDICT:", report["verdict"])
    json.dump(report, open("research_range_horizons_report.json", "w"), indent=2)

if __name__ == "__main__":
    main()
