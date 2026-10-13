"""
research_intraday.py -- RESEARCH ONLY (touches no prediction file).

Question: can gold's direction over the next 5/10/20/30/60 minutes be predicted
better than chance, tested strictly (walk-forward, no look-ahead)?

Data (free, via yfinance; limits are Yahoo's): gold futures GC=F at 5-minute bars
(last ~60 days) and 60-minute bars (last ~2 years), plus cross-asset bars
(dollar DX-Y.NYB, 10y yield ^TNX, silver SI=F, S&P futures ES=F) used as extra
features. Flat bars (zero move) are dropped because they are stale quotes, not signal.

Output: research_intraday_report.json + printed table.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

EXOG = {"usd": "DX-Y.NYB", "y10": "^TNX", "silver": "SI=F", "spx": "ES=F"}
# (bar interval, yahoo period, {label: horizon in bars})
PLANS = [
    ("5m", "60d", {"5min": 1, "10min": 2, "20min": 4, "30min": 6, "60min": 12}),
    ("60m", "730d", {"1h": 1, "2h": 2, "4h": 4}),
]
STEP = 600

def _close(df):
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.astype(float).dropna()
    c.index = pd.to_datetime(c.index)
    return c

def load_live(interval, period):
    import yfinance as yf
    gold = _close(yf.download("GC=F", period=period, interval=interval, progress=False, auto_adjust=True))
    X = {}
    for k, t in EXOG.items():
        try:
            X[k] = _close(yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True))
        except Exception as e:
            print("exog failed", t, e)
    return gold, X

def load_selftest(n=12000, rho=0.04, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    r = np.zeros(n)
    e = rng.normal(0, 0.0005, n)
    for i in range(1, n):
        r[i] = rho * r[i - 1] + e[i]
    gold = pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)
    X = {k: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n))), index=idx) for k in EXOG}
    return gold, X

def features(gold, X):
    f = pd.DataFrame(index=gold.index)
    r = gold.pct_change()
    for k in (1, 2, 3, 6, 12):
        f[f"r{k}"] = gold.pct_change(k)
    f["vol12"], f["vol48"] = r.rolling(12).std(), r.rolling(48).std()
    f["z24"] = (gold - gold.rolling(24).mean()) / (gold.rolling(24).std() + 1e-12)
    d = gold.diff()
    f["rsi14"] = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / ((-d.clip(upper=0)).rolling(14).mean() + 1e-12))
    hr = (gold.index.hour + gold.index.minute / 60.0)
    f["h_sin"], f["h_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["dow"] = gold.index.dayofweek
    for name, s in X.items():
        a = s.reindex(gold.index.union(s.index)).ffill(limit=3).reindex(gold.index)
        f[f"{name}_r1"], f[f"{name}_r3"] = a.pct_change(1), a.pct_change(3)
    return f

def walk_forward(F, gold, H, model):
    fut = gold.shift(-H) / gold - 1
    y = (fut > 0).astype(int)
    valid = F.notna().all(axis=1) & fut.notna() & (fut != 0)       # drop flat/stale bars
    idx = np.where(valid)[0]
    if len(idx) < 3000:
        return None
    start = int(len(idx) * 0.4)
    P, Y, T = [], [], []
    for s in range(start, len(idx), STEP):
        tr = idx[:s]; tr = tr[tr <= idx[s] - H]                      # purge overlapping labels
        te = idx[s:s + STEP]
        sc = StandardScaler().fit(F.iloc[tr])
        m = (LogisticRegression(C=0.05, max_iter=400) if model == "logistic" else
             GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0))
        m.fit(sc.transform(F.iloc[tr]), y.iloc[tr])
        P += list(m.predict_proba(sc.transform(F.iloc[te]))[:, 1]); Y += list(y.iloc[te]); T += list(F.index[te])
    return np.array(P), np.array(Y), pd.DatetimeIndex(T)

def summarise(P, Y, T):
    pred = (P > 0.5).astype(int); n = len(Y); k = int((pred == Y).sum())
    base = max(Y.mean(), 1 - Y.mean())
    out = {"n": n, "accuracy": k / n, "baseline": float(base),
           "p_value": float(stats.binomtest(k, n, base, alternative="greater").pvalue)}
    conf = np.abs(P - 0.5)
    for q in (0.3, 0.1):
        mk = conf >= np.quantile(conf, 1 - q); kk = int((pred[mk] == Y[mk]).sum()); nn = int(mk.sum())
        out[f"top{int(q*100)}"] = {"n": nn, "accuracy": kk / nn,
                                  "p_value_vs_50": float(stats.binomtest(kk, nn, 0.5, alternative="greater").pvalue)}
    # stability: share of ~weekly blocks beating the baseline
    wk = pd.Series(pred == Y, index=T).groupby(pd.Grouper(freq="7D")).agg(["mean", "count"])
    wk = wk[wk["count"] >= 100]
    out["blocks_above_baseline"] = f"{int((wk['mean'] > base).sum())}/{len(wk)}"
    return out

def main():
    selftest = "--selftest" in sys.argv
    plans = [("5m", "synthetic", {"5min": 1, "10min": 2, "30min": 6})] if selftest else PLANS
    report = {"results": {}, "selftest": selftest}
    n_tests = 0
    for interval, period, hz in plans:
        gold, X = load_selftest() if selftest else load_live(interval, period)
        F = features(gold, X)
        print(f"\n[{interval}] bars={len(gold)} {gold.index.min()} -> {gold.index.max()}")
        for label, H in hz.items():
            for model in ("logistic", "gradient_boosting"):
                res = walk_forward(F, gold, H, model)
                if res is None:
                    print(f"  {label:6s} {model:17s} too little data"); continue
                s = summarise(*res); n_tests += 1
                report["results"][f"{interval}|{label}|{model}"] = s
                print(f"  {label:6s} {model:17s} n={s['n']:6d} acc={s['accuracy']:.3f} base={s['baseline']:.3f} "
                      f"p={s['p_value']:.4f} top30={s['top30']['accuracy']:.3f} top10={s['top10']['accuracy']:.3f} blocks>base={s['blocks_above_baseline']}")
    alpha = 0.01 / max(n_tests, 1)                                  # Bonferroni: we tried many combos
    good = []
    for k, v in report["results"].items():
        a, b = v["blocks_above_baseline"].split("/")
        if v["p_value"] < alpha and v["accuracy"] - v["baseline"] >= 0.02 and int(b) > 0 and int(a) / int(b) >= 0.7:
            good.append(k)
    report["tests_run"], report["bonferroni_alpha"], report["credible_edges"] = n_tests, alpha, good
    report["verdict"] = ("Credible short-horizon edge in: " + ", ".join(good)) if good else \
        f"No credible short-horizon edge (needs p<{alpha:.5f} after correcting for {n_tests} tests, +2pts over baseline, 70%+ of weeks)."
    print("\nVERDICT:", report["verdict"])
    json.dump(report, open("research_intraday_report.json", "w"), indent=2)

if __name__ == "__main__":
    main()
