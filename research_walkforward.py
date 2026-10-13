"""
research_walkforward.py -- RESEARCH ONLY. Does not touch any prediction file.

Question it answers honestly: do gold + macro features (real yield, dollar,
VIX, Fed funds, yield slope) give a statistically real edge in predicting
gold direction, tested the strict way (walk-forward, no look-ahead)?

Data (all free, no key needed): gold futures from Yahoo (via yfinance),
macro series from FRED's public CSV endpoint. Macro values are shifted one
trading day because FRED publishes them after the close (no look-ahead).

Output: research_walkforward_report.json + a printed table.
"""
import sys, json, io, warnings
import numpy as np, pandas as pd, requests
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

FRED = {"real_yield": "DFII10", "dollar": "DTWEXBGS", "vix": "VIXCLS",
        "fed_funds": "DFF", "y2": "DGS2", "y10": "DGS10"}
HORIZONS = [1, 5, 10, 20]
STEP = 60

def fred(series):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = ["Date", "v"]
    d["Date"] = pd.to_datetime(d["Date"])
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.set_index("Date")["v"]

def load_live():
    import yfinance as yf
    g = yf.download("GC=F", period="max", interval="1d", progress=False, auto_adjust=True)
    if g is None or len(g) < 1000:
        g = yf.download("GLD", period="max", interval="1d", progress=False, auto_adjust=True)
    close = g["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    df = pd.DataFrame({"gold": close.astype(float)}).dropna()
    for name, sid in FRED.items():
        s = fred(sid)
        df[name] = s.reindex(df.index.union(s.index)).ffill().reindex(df.index).shift(1)  # +1 day lag: no look-ahead
    return df.dropna(subset=["gold"]).ffill()

def load_selftest():
    """Offline smoke test using the bundled 2008-2018 CSV (stand-in macro columns)."""
    d = pd.read_csv("gld_price_data.csv"); d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date").sort_index()
    return pd.DataFrame({"gold": d["GLD"], "real_yield": d["SPX"] / 1000, "dollar": 1 / d["EUR/USD"],
                         "vix": d["USO"], "fed_funds": d["SLV"] / 10, "y2": d["SPX"] / 700, "y10": d["SPX"] / 600})

def features(df, use_macro=True):
    f = pd.DataFrame(index=df.index)
    r = df["gold"].pct_change()
    f["r1"], f["r5"], f["r20"] = r, df["gold"].pct_change(5), df["gold"].pct_change(20)
    f["vol10"] = r.rolling(10).std()
    f["ma_ratio"] = df["gold"].rolling(5).mean() / df["gold"].rolling(20).mean()
    d = df["gold"].diff()
    f["rsi"] = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / ((-d.clip(upper=0)).rolling(14).mean() + 1e-9))
    if use_macro:
        f["ry"], f["ry_d5"] = df["real_yield"], df["real_yield"].diff(5)
        f["usd_r5"], f["usd_r20"] = df["dollar"].pct_change(5), df["dollar"].pct_change(20)
        f["vix"], f["vix_d5"] = df["vix"], df["vix"].diff(5)
        f["ff"] = df["fed_funds"]
        f["slope"] = df["y10"] - df["y2"]
    return f

def walk_forward(F, price, H, model_name):
    y = (price.shift(-H) / price - 1 > 0).astype(int)
    valid = F.notna().all(axis=1) & price.shift(-H).notna()
    idx = np.where(valid)[0]
    start = int(len(idx) * 0.4)
    P, Y, D = [], [], []
    for s in range(start, len(idx), STEP):
        tr = idx[:s]; tr = tr[tr <= idx[s] - H]      # purge overlapping labels
        te = idx[s:s + STEP]
        sc = StandardScaler().fit(F.iloc[tr])
        m = (LogisticRegression(C=0.1, max_iter=500) if model_name == "logistic"
             else GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0))
        m.fit(sc.transform(F.iloc[tr]), y.iloc[tr])
        P += list(m.predict_proba(sc.transform(F.iloc[te]))[:, 1]); Y += list(y.iloc[te]); D += list(F.index[te])
    return np.array(P), np.array(Y), pd.DatetimeIndex(D)

def summarise(P, Y, D):
    pred = (P > 0.5).astype(int); n = len(Y); k = int((pred == Y).sum())
    base_rate = max(Y.mean(), 1 - Y.mean())
    out = {"n": n, "accuracy": k / n, "majority_baseline": float(base_rate),
           "p_value_vs_baseline": float(stats.binomtest(k, n, base_rate, alternative="greater").pvalue)}
    conf = np.abs(P - 0.5)
    for q in (0.3, 0.1):
        mk = conf >= np.quantile(conf, 1 - q); kk = int((pred[mk] == Y[mk]).sum()); nn = int(mk.sum())
        out[f"top{int(q*100)}pct"] = {"n": nn, "accuracy": kk / nn,
                                     "p_value_vs_50": float(stats.binomtest(kk, nn, 0.5, alternative="greater").pvalue)}
    yrs = pd.Series(pred == Y, index=D).groupby(D.year).mean()
    out["by_year"] = {int(a): round(float(b), 3) for a, b in yrs.items()}
    out["years_above_baseline"] = f"{int((yrs > base_rate).sum())}/{len(yrs)}"
    return out

def main():
    selftest = "--selftest" in sys.argv
    df = load_selftest() if selftest else load_live()
    print(f"gold rows: {len(df)}  {df.index.min().date()} -> {df.index.max().date()}  (selftest={selftest})")
    report = {"rows": len(df), "start": str(df.index.min().date()), "end": str(df.index.max().date()), "results": {}}
    for label, use_macro in (("price_only", False), ("price_plus_macro", True)):
        F = features(df, use_macro)
        for H in HORIZONS:
            for mn in ("logistic", "gradient_boosting"):
                P, Y, D = walk_forward(F, df["gold"], H, mn)
                s = summarise(P, Y, D); report["results"][f"{label}|{H}d|{mn}"] = s
                print(f"{label:17s} H={H:2d}d {mn:17s} n={s['n']:5d} acc={s['accuracy']:.3f} base={s['majority_baseline']:.3f} "
                      f"p={s['p_value_vs_baseline']:.3f} top30={s['top30pct']['accuracy']:.3f} yrs>base={s['years_above_baseline']}")
    # Verdict: a real edge needs significance (p<0.01 after trying many combos), >=3pts over baseline, and most years above baseline.
    best = [(k, v) for k, v in report["results"].items()
            if v["p_value_vs_baseline"] < 0.01 and v["accuracy"] - v["majority_baseline"] >= 0.03
            and int(v["years_above_baseline"].split("/")[0]) / int(v["years_above_baseline"].split("/")[1]) >= 0.7]
    report["credible_edges"] = [k for k, _ in best]
    report["verdict"] = ("Credible edge found in: " + ", ".join(report["credible_edges"])) if best else \
        "No credible, stable directional edge found (strict threshold: p<0.01, +3pts over baseline, 70%+ of years)."
    print("\nVERDICT:", report["verdict"])
    json.dump(report, open("research_walkforward_report.json", "w"), indent=2)

if __name__ == "__main__":
    main()
