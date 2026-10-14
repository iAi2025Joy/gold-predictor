"""
research_neural.py -- RESEARCH ONLY (touches no prediction file).

Question: can a small NEURAL NETWORK predict the 10 / 20 / 30-minute direction of gold (GC=F) and oil (CL=F)
better than chance -- and better than a plain logistic regression -- when tested strictly walk-forward?

How the test is kept honest
  * Walk-forward: the model is trained only on the past and scored on the next unseen block (600 bars), then
    the window moves forward. Labels that overlap the test start are purged.
  * Flat bars (no price change) are dropped. Same rules as research_intraday.py.
  * The bar of the call is the last COMPLETED bar; only information up to it is used.
  * Bonferroni: every asset x horizon x model combination counts as one test; the credible bar is p < 0.01 / tests.
  * A network must also beat the logistic baseline by 1+ point and win 70%+ of weekly blocks.
  * Self-test (python research_neural.py --selftest): a pure-noise series must give NO credible edge, and a series
    with a hidden NON-LINEAR rule must be found by the network but not by logistic regression.

Data (free, via yfinance; Yahoo gives ~60 days of 5-minute bars). Output: research_neural_report.json.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ASSETS = {"gold": "GC=F", "oil": "CL=F"}
EXOG = {"usd": "DX-Y.NYB", "y10": "^TNX", "spx": "ES=F"}
HORIZONS = {"10min": 2, "20min": 4, "30min": 6}
STEP = 600
SEEDS = (0, 1, 2)          # the network is averaged over 3 random starts to reduce luck

def _close(df):
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.astype(float).dropna()
    c.index = pd.to_datetime(c.index)
    return c

def load_live():
    import yfinance as yf
    X = {}
    for k, t in EXOG.items():
        try:
            X[k] = _close(yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True))
        except Exception as e:
            print("exog failed", t, e)
    A = {a: _close(yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True)) for a, t in ASSETS.items()}
    return A, X

def features(p, X):
    f = pd.DataFrame(index=p.index)
    r = p.pct_change()
    for k in range(1, 13):                       # the last 12 individual 5-minute returns (what the network can use non-linearly)
        f[f"lag{k}"] = r.shift(k - 1)
    for k in (3, 6, 12, 24):
        f[f"r{k}"] = p.pct_change(k)
    f["vol12"], f["vol48"] = r.rolling(12).std(), r.rolling(48).std()
    f["z24"] = (p - p.rolling(24).mean()) / (p.rolling(24).std() + 1e-12)
    f["spike"] = (r.abs() / (r.rolling(96).std() + 1e-12)).clip(upper=8)
    ny = p.index.tz_convert("America/New_York") if p.index.tz is not None else p.index
    hr = ny.hour + ny.minute / 60.0
    f["h_sin"], f["h_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["event_bar"] = [1.0 if (h in (8.5, 9.5, 10.0)) else 0.0 for h in hr]       # 8:30 / 9:30 / 10:00 New York releases
    for name, s in X.items():
        a = s.reindex(p.index.union(s.index)).ffill(limit=3).reindex(p.index)
        f[f"{name}_r1"], f[f"{name}_r3"] = a.pct_change(1), a.pct_change(3)
    return f

def make_model(kind, seed=0):
    if kind == "logistic":
        return LogisticRegression(C=0.05, max_iter=400)
    return MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=60,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=6, random_state=seed)

def walk_forward(F, p, H, kind):
    fut = p.shift(-H) / p - 1
    y = (fut > 0).astype(int)
    valid = F.notna().all(axis=1) & fut.notna() & (fut != 0)
    idx = np.where(valid)[0]
    if len(idx) < 3000:
        return None
    start = int(len(idx) * 0.4)
    P, Y, T = [], [], []
    for s in range(start, len(idx), STEP):
        tr = idx[:s]; tr = tr[tr <= idx[s] - H]
        te = idx[s:s + STEP]
        sc = StandardScaler().fit(F.iloc[tr])
        Xtr, Xte = sc.transform(F.iloc[tr]), sc.transform(F.iloc[te])
        seeds = SEEDS if kind == "neural" else (0,)
        pr = np.mean([make_model(kind, sd).fit(Xtr, y.iloc[tr]).predict_proba(Xte)[:, 1] for sd in seeds], axis=0)
        P += list(pr); Y += list(y.iloc[te]); T += list(F.index[te])
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
    wk = pd.Series(pred == Y, index=T).groupby(pd.Grouper(freq="7D")).agg(["mean", "count"])
    wk = wk[wk["count"] >= 100]
    out["blocks_above_baseline"] = f"{int((wk['mean'] > base).sum())}/{len(wk)}"
    return out

# ------------------------------------------------------------------ self-test data
def synth(kind, n=9000, seed=3):
    """kind='noise': random walk. kind='hidden': the next move's SIGN is the product of the signs of the last two moves
    (a non-linear XOR rule that a straight-line model cannot see) with 70% reliability."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    r = rng.normal(0, 0.0005, n)
    if kind == "hidden":
        for i in range(3, n):
            s = np.sign(r[i - 1]) * np.sign(r[i - 2])
            if rng.random() < 0.70:
                r[i] = s * abs(r[i])
            else:
                r[i] = -s * abs(r[i])
    p = pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)
    X = {k: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n))), index=idx) for k in EXOG}
    return {"gold": p}, X

def run(A, X, hz):
    results = {}
    for asset, p in A.items():
        F = features(p, X)
        print(f"\n[{asset}] bars={len(p)} {p.index.min()} -> {p.index.max()}")
        for label, H in hz.items():
            for kind in ("logistic", "neural"):
                res = walk_forward(F, p, H, kind)
                if res is None:
                    print(f"  {label:6s} {kind:9s} too little data"); continue
                s = summarise(*res); results[f"{asset}|{label}|{kind}"] = s
                print(f"  {label:6s} {kind:9s} n={s['n']:6d} acc={s['accuracy']:.3f} base={s['baseline']:.3f} p={s['p_value']:.4f} "
                      f"top30={s['top30']['accuracy']:.3f} top10={s['top10']['accuracy']:.3f} blocks>base={s['blocks_above_baseline']}")
    return results

def verdict(results):
    n_tests = len(results); alpha = 0.01 / max(n_tests, 1); good = []
    for k, v in results.items():
        if not k.endswith("|neural"):
            continue
        a, b = v["blocks_above_baseline"].split("/")
        lg = results.get(k.replace("|neural", "|logistic"))
        if (v["p_value"] < alpha and v["accuracy"] - v["baseline"] >= 0.02 and int(b) > 0 and int(a) / int(b) >= 0.7
                and (lg is None or v["accuracy"] - lg["accuracy"] >= 0.01)):
            good.append(k)
    text = ("Neural network shows a credible edge in: " + ", ".join(good)) if good else \
        f"No credible neural-network edge (needs p<{alpha:.5f} after correcting for {n_tests} tests, +2 points over the baseline, 70%+ of weeks, and +1 point over logistic)."
    return n_tests, alpha, good, text

def main():
    if "--selftest" in sys.argv:
        ok = True
        for kind in ("noise", "hidden"):
            A, X = synth(kind)
            res = run(A, X, {"5min": 1})
            n, alpha, good, text = verdict(res)
            lg = res["gold|5min|logistic"]["accuracy"]; nn = res["gold|5min|neural"]["accuracy"]
            print(f"SELFTEST {kind}: logistic={lg:.3f} neural={nn:.3f} -> {text}")
            if kind == "noise" and good:
                ok = False; print("FAIL: noise must give no credible edge")
            if kind == "hidden" and not (nn > 0.60 and nn > lg + 0.03 and good):
                ok = False; print("FAIL: hidden non-linear rule should be found by the network only")
        print("SELFTEST PASSED" if ok else "SELFTEST FAILED"); sys.exit(0 if ok else 1)
    A, X = load_live()
    res = run(A, X, HORIZONS)
    n, alpha, good, text = verdict(res)
    print("\nVERDICT:", text)
    json.dump({"results": res, "tests_run": n, "bonferroni_alpha": alpha, "credible_neural_edges": good, "verdict": text},
              open("research_neural_report.json", "w"), indent=2)

if __name__ == "__main__":
    main()
