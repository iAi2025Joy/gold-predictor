"""
research_conditions.py -- RESEARCH ONLY (touches no prediction file).

Question: is there ANY recognisable market condition in which the 10 / 20 / 30-minute direction of gold or oil is predictable
better than chance -- so that GARNET could give a direction only in those moments and say "no call" the rest of the time?

Design (built to avoid fooling ourselves):
  * Conditions are a fixed menu made BEFORE looking at results: for each of 5 measures of recent movement (last 1, 3, 6, 12 bars and
    distance from the 2-hour average, each in units of normal volatility), 3 thresholds (1.0 / 1.5 / 2.0), 2 behaviours
    (follow the move / fade the move) and 4 sessions (all day, Asia, London, New York) = 120 conditions.
  * DISCOVERY = first 60% of the 60 days. A condition survives only with >= 100 independent samples, accuracy >= 55% and p < 0.01.
  * HOLD-OUT = last 40%, never used for choosing. A survivor is VALIDATED only if on the hold-out it beats 50% and the commoner-direction
    baseline with p < 0.05 AFTER a Bonferroni correction for the number of survivors, and accuracy is >= 53%.
  * Overlapping horizons are handled by counting independent samples as n / horizon-in-bars.
  * Self-test: pure noise must validate nothing; a hidden rule (fade after a >2-sigma 3-bar move, 65% reliable) must be found.
Output: research_conditions_report.json.   Run: python research_conditions.py      Self-test: python research_conditions.py --selftest
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ASSETS = {"gold": "GC=F", "oil": "CL=F"}
HORIZONS = {"10min": 2, "20min": 4, "30min": 6}
MEASURES = ["m1", "m3", "m6", "m12", "dist24"]
THRESHOLDS = (1.0, 1.5, 2.0)
SESSIONS = {"all": (0, 24), "asia": (0, 7), "london": (7, 13), "ny": (13, 21)}
DISC_FRAC = 0.6
MIN_INDEP, DISC_ACC, DISC_P = 100, 0.55, 0.01
HOLD_ACC, HOLD_ALPHA = 0.53, 0.05

def load_live():
    import yfinance as yf
    out = {}
    for a, t in ASSETS.items():
        df = yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True)
        c = df["Close"]
        c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        c = c.astype(float).dropna(); c.index = pd.to_datetime(c.index)
        out[a] = c
    return out

def measures(p):
    lp = np.log(p.values.astype(float))
    r1 = np.diff(lp, prepend=np.nan)
    vol = pd.Series(r1).rolling(96).std().shift(1).values            # normal 5-minute volatility, known before the bar
    M = {}
    for k, name in ((1, "m1"), (3, "m3"), (6, "m6"), (12, "m12")):
        M[name] = (lp - pd.Series(lp).shift(k).values) / (vol * np.sqrt(k))
    ma = pd.Series(lp).rolling(24).mean().values
    sd = pd.Series(lp).rolling(24).std().values
    M["dist24"] = (lp - ma) / (sd + 1e-12)
    return M, lp

def hours_utc(idx):
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx
    return (idx.hour + idx.minute / 60.0).values

def evaluate(p, H, lo, hi, cond=None):
    """Returns per-condition arrays over bars [lo, hi)."""
    M, lp = measures(p)
    fut = np.full(len(lp), np.nan); fut[:-H] = lp[H:] - lp[:-H]
    hr = hours_utc(p.index)
    res = {}
    for meas in MEASURES:
        f = M[meas]
        for thr in THRESHOLDS:
            for sess, (a, b) in SESSIONS.items():
                sel = np.zeros(len(lp), bool); sel[lo:hi] = True
                sel &= np.isfinite(f) & np.isfinite(fut) & (fut != 0) & (np.abs(f) >= thr) & (hr >= a) & (hr < b)
                idx = np.where(sel)[0]
                if len(idx) == 0:
                    continue
                for mode in ("follow", "fade"):
                    pred_up = (f[idx] > 0) if mode == "follow" else (f[idx] < 0)
                    real_up = fut[idx] > 0
                    correct = pred_up == real_up
                    res[(meas, thr, sess, mode)] = (int(correct.sum()), len(idx), float(real_up.mean()), len(idx) / max(1, len(lp[lo:hi])))
    return res

def run(prices, horizons):
    report = {"assets": {}, "design": {"discovery_fraction": DISC_FRAC, "conditions": len(MEASURES) * len(THRESHOLDS) * len(SESSIONS) * 2}}
    validated_all = []
    for asset, p in prices.items():
        n = len(p); cut = int(n * DISC_FRAC)
        report["assets"][asset] = {}
        for label, H in horizons.items():
            disc = evaluate(p, H, 100, cut)
            hold = evaluate(p, H, cut, n - H)
            surv = []
            for key, (k, m, up, cov) in disc.items():
                n_eff = m / H
                if n_eff < MIN_INDEP or k / m < DISC_ACC:
                    continue
                if stats.binomtest(int(round(k / H)), int(round(n_eff)), 0.5, alternative="greater").pvalue < DISC_P:
                    surv.append(key)
            rows = []
            for key in surv:
                k, m, up, cov = hold.get(key, (0, 0, 0.5, 0))
                n_eff = int(round(m / H)); k_eff = int(round(k / H))
                if n_eff < 30:
                    continue
                base = max(up, 1 - up)
                pval = stats.binomtest(k_eff, n_eff, max(0.5, base), alternative="greater").pvalue
                pcorr = min(1.0, pval * max(1, len(surv)))
                ok = (k / m >= HOLD_ACC and pcorr < HOLD_ALPHA)
                rows.append({"condition": f"{key[0]} >= {key[1]} sigma, {key[2]}, {key[3]}", "disc_accuracy": round(disc[key][0] / disc[key][1], 3),
                             "hold_accuracy": round(k / m, 3), "hold_baseline": round(base, 3), "hold_independent_n": n_eff,
                             "share_of_time": round(cov, 4), "p_corrected": round(float(pcorr), 4), "validated": bool(ok)})
                if ok:
                    validated_all.append(f"{asset}|{label}|{rows[-1]['condition']}")
            rows.sort(key=lambda r: -r["hold_accuracy"])
            report["assets"][asset][label] = {"conditions_tested": len(disc), "survived_discovery": len(surv), "holdout": rows[:12]}
            print(f"[{asset} {label}] tested={len(disc)} survived discovery={len(surv)} validated on hold-out={sum(r['validated'] for r in rows)}")
            for r in rows[:5]:
                print("   ", r["condition"], "disc", r["disc_accuracy"], "hold", r["hold_accuracy"], "base", r["hold_baseline"], "n", r["hold_independent_n"], "OK" if r["validated"] else "")
    report["validated"] = validated_all
    report["verdict"] = ("Conditions with a validated edge: " + "; ".join(validated_all)) if validated_all else \
        "No condition validated on the hold-out period: no moment found in which direction is reliably predictable."
    return report

def synth(kind, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    r = rng.normal(0, 0.0005, n)
    if kind == "hidden":                                        # fade after a >2-sigma 3-bar move, 65% reliable, only in the NY session
        sd = 0.0005
        for i in range(12, n - 8):
            m3 = r[i - 2:i + 1].sum() / (sd * np.sqrt(3))
            hh = idx[i].hour
            if abs(m3) > 2.0 and 13 <= hh < 21:
                s = -np.sign(m3)
                r[i + 1] = (s if rng.random() < 0.65 else -s) * abs(r[i + 1]) * 1.5
    return {"gold": pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)}

def main():
    if "--selftest" in sys.argv:
        ok = True
        for kind in ("noise", "hidden"):
            rep = run(synth(kind), {"5min": 1, "10min": 2})
            print(f"SELFTEST {kind}: {rep['verdict']}")
            if kind == "noise" and rep["validated"]: ok = False; print("FAIL: noise validated something")
            if kind == "hidden" and not rep["validated"]: ok = False; print("FAIL: hidden rule not found")
        print("SELFTEST PASSED" if ok else "SELFTEST FAILED"); sys.exit(0 if ok else 1)
    rep = run(load_live(), HORIZONS)
    print("\nVERDICT:", rep["verdict"])
    json.dump(rep, open("research_conditions_report.json", "w"), indent=2)

if __name__ == "__main__":
    main()
