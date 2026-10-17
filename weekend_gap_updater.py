"""Weekend / closed-market gap statistics for gold and oil.
Measures, over about two years of hourly futures bars, how far the price moved between the last trade before a closure
(weekend or holiday, any gap longer than 40 hours) and the first trade after the reopening. Writes weekend_gap.json,
which GARNET uses to answer "what will the price be when the market reopens" while the market is closed."""
import json, sys, datetime as dt
import numpy as np, pandas as pd

OUT = "weekend_gap.json"
TICKERS = {"gold": "GC=F", "oil": "CL=F"}
GAP_HOURS = 40


def gap_stats(close):
    close = close.dropna()
    close = close[close > 0]
    gaps = close.index.to_series().diff() > pd.Timedelta(hours=GAP_HOURS)
    idx = np.where(gaps.values)[0]
    r = np.array([np.log(close.iloc[i] / close.iloc[i - 1]) for i in idx if i > 0])
    if len(r) < 20:
        return None
    q = np.percentile(r, [5, 15, 50, 85, 95])
    return {"n_gaps": int(len(r)), "q05": float(q[0]), "q15": float(q[1]), "q50": float(q[2]), "q85": float(q[3]),
            "q95": float(q[4]), "share_up": float((r > 0).mean()), "typical_abs_pct": float(np.median(np.abs(r)) * 100),
            "first_gap": str(close.index[idx[0]].date()), "last_gap": str(close.index[idx[-1]].date())}


def main():
    import yfinance as yf
    out = {"updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "assets": {}}
    for name, t in TICKERS.items():
        d = yf.download(t, period="730d", interval="1h", progress=False, auto_adjust=True)
        c = d["Close"]
        c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        s = gap_stats(c.astype(float))
        if s:
            out["assets"][name] = s
    if not out["assets"]:
        print("no data"); sys.exit(1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


def selftest():
    rng = np.random.default_rng(1)
    idx = []
    t = pd.Timestamp("2025-01-06 00:00", tz="UTC")
    for w in range(100):
        for h in range(24 * 5):
            idx.append(t + pd.Timedelta(hours=h))
        t += pd.Timedelta(days=7)
    steps = rng.normal(0, 0.002, len(idx))
    c = pd.Series(2000 * np.exp(np.cumsum(steps)), index=pd.DatetimeIndex(idx))
    s = gap_stats(c)
    assert s and 95 <= s["n_gaps"] <= 100, s
    assert s["q05"] < s["q50"] < s["q95"]
    print("selftest ok", s["n_gaps"])


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
