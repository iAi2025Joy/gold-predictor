"""
research_analogs.py -- RESEARCH ONLY. Touches no prediction file.

Idea (yours): predict the next 10-60 minutes by finding SIMILAR past situations and seeing how price moved afterwards.
This is "analog forecasting" (k-nearest-neighbours on market situations). Tested strictly: at every point only PAST
situations whose outcome was already known are searched, so nothing leaks from the future.

Situation features: recent moves (1,2,3,6,12 bars, scaled by volatility), short-vs-long volatility, position vs recent mean,
time of day, and cross-asset moves (dollar, 10y yield, silver, S&P futures).
Output per horizon: accuracy vs the majority baseline, p-value (Bonferroni-corrected for all variants tried), Brier score
(is the stated probability honest?), accuracy when the analogs are strongly one-sided, and week-by-week stability.
Data: Yahoo 5-minute bars (~60 days) for gold GC=F and oil CL=F.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ASSETS = {"gold": "GC=F", "oil": "CL=F"}
EXOG = {"usd": "DX-Y.NYB", "y10": "^TNX", "silver": "SI=F", "spx": "ES=F"}
HORIZONS_MIN = [10, 20, 30, 60]
KS = [50, 200]
WARMUP, STRIDE_MIN = 2500, 2
CORE = ["r1", "r2", "r3", "r6", "r12"]      # only the recent-move shape (fewer features = analogs are not diluted)
FEATSET = "full"


def _close(d):
    c = d["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    c = c.astype(float).dropna(); c.index = pd.to_datetime(c.index)
    return c[c > 0]


def load_live(ticker):
    import yfinance as yf
    g = _close(yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True))
    X = {}
    for k, t in EXOG.items():
        try: X[k] = _close(yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True))
        except Exception as e: print("exog failed", t, e)
    return g, X


def load_selftest(seed=7, n=14000, rho=0.12):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    r = np.zeros(n); e = rng.standard_t(6, n) * 0.0005
    for i in range(1, n): r[i] = rho * r[i - 1] + e[i]            # injected short-term momentum: analogs SHOULD find it
    g = pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)
    X = {k: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n))), index=idx) for k in EXOG}
    return g, X


def situation_features(g, X):
    lp = np.log(g); r1 = lp.diff()
    vol = r1.rolling(36).std()
    f = pd.DataFrame(index=g.index)
    for k in (1, 2, 3, 6, 12):
        f[f"r{k}"] = (lp - lp.shift(k)) / (vol * np.sqrt(k))
    f["volratio"] = r1.rolling(6).std() / r1.rolling(72).std()
    f["z24"] = (g - g.rolling(24).mean()) / (g.rolling(24).std() + 1e-12)
    h = g.index.hour + g.index.minute / 60.0
    f["h_sin"], f["h_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    for name, s in X.items():
        a = np.log(s.reindex(g.index.union(s.index)).ffill(limit=3).reindex(g.index))
        for k in (1, 3):
            f[f"{name}{k}"] = (a - a.shift(k)) / (a.diff().rolling(36).std() * np.sqrt(k))
    return f.replace([np.inf, -np.inf], np.nan)


def session(ts_index):
    h = ts_index.hour
    return np.where(h < 7, 0, np.where(h < 13, 1, 2))


def analog_eval(g, F, H, K, same_session):
    F = F[CORE] if FEATSET == "core" else F
    lp = np.log(g.values); n = len(lp)
    fwd = np.full(n, np.nan); fwd[:n - H] = lp[H:] - lp[:-H]
    ok = F.notna().all(axis=1).values & np.isfinite(fwd) & (fwd != 0)
    Fv = F.values; sess = session(g.index)
    idx_all = np.where(ok)[0]
    probs, ys, ts = [], [], []
    for t in range(WARMUP, n - H, STRIDE_MIN * H):
        if not (np.isfinite(Fv[t]).all()): continue
        past = idx_all[idx_all <= t - H]                                # outcome fully known at time t
        if same_session: past = past[sess[past] == sess[t]]
        if len(past) < 400: continue
        mu, sd = Fv[past].mean(0), Fv[past].std(0) + 1e-9               # standardise with PAST stats only
        d = (((Fv[past] - mu) / sd - (Fv[t] - mu) / sd) ** 2).sum(1)
        nn = past[np.argpartition(d, K)[:K]]
        probs.append(float((fwd[nn] > 0).mean())); ys.append(int(fwd[t] > 0)); ts.append(g.index[t])
    return np.array(probs), np.array(ys), pd.DatetimeIndex(ts)


def summarise(p, y, ts):
    pred = (p > 0.5).astype(int); n = len(y); k = int((pred == y).sum())
    base = max(y.mean(), 1 - y.mean())
    out = {"n": n, "accuracy": k / n, "baseline": float(base),
           "p_value": float(stats.binomtest(k, n, base, alternative="greater").pvalue),
           "brier": float(np.mean((p - y) ** 2)), "brier_naive": float(np.mean((y.mean() - y) ** 2))}
    strong = (p >= 0.62) | (p <= 0.38); ns = int(strong.sum())
    if ns >= 40:
        ks = int((pred[strong] == y[strong]).sum())
        out["strong"] = {"n": ns, "accuracy": ks / ns, "p_value_vs_50": float(stats.binomtest(ks, ns, 0.5, alternative="greater").pvalue)}
    wk = pd.Series(pred == y, index=ts).groupby(pd.Grouper(freq="7D")).agg(["mean", "count"]); wk = wk[wk["count"] >= 60]
    out["weeks_above_baseline"] = f"{int((wk['mean'] > base).sum())}/{len(wk)}"
    return out


def main():
    selftest = "--selftest" in sys.argv
    res, ntests = {}, 0
    for name, tk in ASSETS.items():
        if selftest and name == "oil": continue
        g, X = load_selftest() if selftest else load_live(tk)
        F = situation_features(g, X)
        print(f"\n{name} ({tk}): {len(g)} bars, {F.shape[1]} situation features")
        for m in HORIZONS_MIN:
            H = m // 5
            for fs in ("full", "core"):
                globals()["FEATSET"] = fs
                for K in KS:
                    p, y, ts = analog_eval(g, F, H, K, False)
                    if len(y) < 200: print(f"  {m}min {fs} K={K}: too little data"); continue
                    s = summarise(p, y, ts); ntests += 1
                    res[f"{name}|{m}min|{fs}|K{K}"] = s
                    st = s.get("strong"); strong_txt = f" strong n={st['n']} acc={st['accuracy']:.3f}" if st else ""
                    print(f"  {m:3d}min {fs:4s} K={K:3d} n={s['n']} acc={s['accuracy']:.3f} base={s['baseline']:.3f} "
                          f"p={s['p_value']:.4f} brier={s['brier']:.4f}(naive {s['brier_naive']:.4f}){strong_txt} wk={s['weeks_above_baseline']}")
    alpha = 0.01 / max(ntests, 1)
    good = []
    for k, v in res.items():
        a, b = v["weeks_above_baseline"].split("/")
        if v["p_value"] < alpha and v["accuracy"] - v["baseline"] >= 0.02 and v["brier"] < v["brier_naive"] and int(b) and int(a) / int(b) >= 0.7:
            good.append(k)
    verdict = ("Credible analog edge in: " + ", ".join(good)) if good else \
        f"No credible analog edge (needs p<{alpha:.5f} after correcting for {ntests} tests, +2pts over baseline, better Brier than naive, 70%+ of weeks)."
    print("\nVERDICT:", verdict)
    json.dump({"tests": ntests, "alpha": alpha, "results": res, "credible": good, "verdict": verdict}, open("research_analogs_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
