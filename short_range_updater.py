"""
short_range_updater.py -- calibrated 70% price ranges for the next 10 / 20 / 30 / 60 minutes
for gold (GC=F) and oil (CL=F), with every range scored live against the real price.

WHY RANGES, NOT DIRECTION: strict walk-forward studies (research_walkforward.py,
research_intraday.py) found no reliable directional edge at any horizon. A price RANGE
can be calibrated, and the honest accuracy number is how often the real price landed
inside it -- measured here, live, for every horizon.

METHOD (validated by research_range_horizons.py on ~60 days of real 5-minute bars):
  range = base_price * exp( z_quantile * vol_5min * sqrt(bars) ), where vol_5min is the
  recent 5-minute volatility and the z-quantiles (15th / 85th percentile) come from the
  last ~600 standardized past moves. Oil uses only past moves from the SAME trading
  session (Asia / Europe / US), because oil's volatility differs a lot by session and
  a single setting was too narrow in Asia/Europe and too wide in the US session.
  Oil measured slightly under target in back-testing (66-69%), so it is calibrated to a
  slightly higher NOMINAL level; the LIVE hit rate (range_live_coverage) is the number
  to trust and to publish.

Output: short_range_latest.json (+ short_range_track_record.json, the live scorecard).
"""
import os, sys, json, warnings
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

LATEST_FILE = "short_range_latest.json"
TRACK_FILE = "short_range_track_record.json"
ASSETS = {"gold": "GC=F", "oil": "CL=F"}
HORIZONS_MIN = [10, 20, 30, 60]
BAR_MIN = 5
TARGET = 0.70
# nominal coverage actually requested from the quantiles (oil measured ~2-4 pts under target)
NOMINAL = {"gold": 0.70, "oil": 0.72}
SESSION_AWARE = {"gold": False, "oil": True}
VOL_BARS, CAL_POINTS, MIN_Z = 36, 600, 150
MAX_TRACK = 3200
LIVE_WINDOW = 300


def session(ts):
    h = ts.hour
    return "asia" if h < 7 else ("europe" if h < 13 else "us")


def fetch_bars(ticker, now):
    import yfinance as yf
    d = yf.download(ticker, period="7d", interval="5m", progress=False, auto_adjust=True)
    c = d["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.astype(float).dropna()
    c.index = pd.to_datetime(c.index)
    c.index = c.index.tz_localize("UTC") if c.index.tz is None else c.index.tz_convert("UTC")
    c = c[c > 0]
    # drop a still-forming bar so the base price is a completed bar's close
    if len(c) and c.index[-1] + timedelta(minutes=BAR_MIN) > now:
        c = c.iloc[:-1]
    return c


def make_range(price, H, asset):
    """Calibrated range for H bars ahead from the last bar of `price`. None if not enough data."""
    lp = np.log(price.values)
    n = len(lp)
    if n < CAL_POINTS + VOL_BARS + H + 5:
        return None
    r1 = np.diff(lp, prepend=np.nan)
    vol = pd.Series(r1).rolling(VOL_BARS).std().values
    t = n - 1
    if not np.isfinite(vol[t]) or vol[t] <= 0:
        return None
    js = np.arange(t - CAL_POINTS - H, t - H + 1)                    # moves fully known at time t
    js = js[js >= 0]
    rH = lp[js + H] - lp[js]
    z = rH / (vol[js] * np.sqrt(H))
    z = z[np.isfinite(z)]
    method = "global"
    if SESSION_AWARE[asset]:
        sess = np.array([session(price.index[j]) for j in js])
        zs = (rH / (vol[js] * np.sqrt(H)))[sess == session(price.index[t])]
        zs = zs[np.isfinite(zs)]
        if len(zs) >= MIN_Z:
            z, method = zs, "same_session"
    if len(z) < MIN_Z:
        return None
    nom = NOMINAL[asset]
    z_lo, z_hi = np.quantile(z, [(1 - nom) / 2, 1 - (1 - nom) / 2])
    base = float(price.iloc[-1]); s = vol[t] * np.sqrt(H)
    return {"range_low": round(base * float(np.exp(z_lo * s)), 4), "range_high": round(base * float(np.exp(z_hi * s)), 4),
            "base_price": base, "method": method, "n_calibration": int(len(z))}


def score_pending(track, price_by_asset, now):
    """Score each pending range against the real bar at its target time."""
    for e in track:
        if e.get("hit") is not None or e.get("expired"):
            continue
        p = price_by_asset.get(e["asset"])
        if p is None or not len(p):
            continue
        tgt = pd.Timestamp(e["target_time"])
        if p.index[-1] < tgt:
            continue                                                  # target not reached yet
        if tgt in p.index:
            actual = float(p.loc[tgt])
        else:
            nxt = p[p.index >= tgt]
            if len(nxt) and (nxt.index[0] - tgt) <= pd.Timedelta(minutes=10):
                actual = float(nxt.iloc[0])
            else:
                e["expired"] = True                                   # market closed / data gap: never score fakes
                continue
        e["actual_price"] = actual
        e["hit"] = bool(e["range_low"] <= actual <= e["range_high"])
    return track


def live_coverage(track, asset, horizon):
    h = [e["hit"] for e in track if e["asset"] == asset and e["horizon_min"] == horizon and e.get("hit") is not None][-LIVE_WINDOW:]
    n = len(h)
    if n == 0:
        return {"live_coverage": None, "live_n": 0, "live_note": "No scored live ranges yet."}
    k = int(sum(h)); c = k / n
    if n < 30:
        note = f"{k}/{n} scored so far -- too few to judge."
    else:
        p = stats.binomtest(k, n, TARGET).pvalue
        note = f"{c:.1%} of the last {n} ranges contained the real price (target {TARGET:.0%}). " + \
               ("Consistent with target." if p >= 0.01 else "Meaningfully OFF target -- treat with caution.")
    return {"live_coverage": round(c, 4), "live_n": n, "live_note": note}


def run_once(price_by_asset, track, now):
    track = score_pending(track, price_by_asset, now)
    existing = {(e["asset"], e["horizon_min"], e["base_time"]) for e in track}
    out = {"updated_at": now.isoformat(), "coverage_target": TARGET, "assets": {},
           "method_note": ("Calibrated price ranges, NOT direction forecasts: direction cannot be predicted better than chance "
                           "(strict walk-forward studies). Each range says where the price is likely to be at the horizon; "
                           "live_coverage is how often real prices actually landed inside these ranges."),
           "data_note": "Based on Yahoo Finance futures 5-minute bars (GC=F, CL=F), which can be delayed several minutes; "
                        "a range is valid for its horizon counted from data_as_of."}
    for asset, price in price_by_asset.items():
        if price is None or not len(price):
            out["assets"][asset] = {"status": "no_data"}; continue
        base_time = price.index[-1]
        entry = {"status": "ok", "data_as_of": base_time.isoformat(), "base_price": float(price.iloc[-1]), "ranges": {}}
        stale_min = (now - base_time).total_seconds() / 60.0
        entry["data_age_minutes"] = round(stale_min, 1)
        if stale_min > 30:
            entry["status"] = "stale"; entry["note"] = "Market data is more than 30 minutes old (market closed or feed delayed); ranges not issued."
            out["assets"][asset] = entry; continue
        for m in HORIZONS_MIN:
            H = m // BAR_MIN
            rg = make_range(price, H, asset)
            if rg is None:
                entry["ranges"][f"{m}min"] = {"status": "insufficient_data"}; continue
            tt = base_time + timedelta(minutes=m)
            key = (asset, m, base_time.isoformat())
            if key not in existing:
                track.append({"asset": asset, "horizon_min": m, "base_time": base_time.isoformat(),
                              "target_time": tt.isoformat(), "base_price": rg["base_price"],
                              "range_low": rg["range_low"], "range_high": rg["range_high"],
                              "method": rg["method"], "hit": None})
            entry["ranges"][f"{m}min"] = {"range_low": rg["range_low"], "range_high": rg["range_high"],
                                          "valid_until": tt.isoformat(), "method": rg["method"], **live_coverage(track, asset, m)}
        out["assets"][asset] = entry
    return out, track[-MAX_TRACK:]


def main():
    now = datetime.now(timezone.utc)
    prices, ok = {}, 0
    for a, tk in ASSETS.items():
        try:
            prices[a] = fetch_bars(tk, now); ok += 1
            print(f"{a}: {len(prices[a])} bars, last {prices[a].index[-1]}")
        except Exception as ex:
            print(f"{a}: fetch failed: {ex}"); prices[a] = None
    if ok == 0:
        print("All fetches failed."); sys.exit(1)
    track = json.load(open(TRACK_FILE)) if os.path.exists(TRACK_FILE) else []
    out, track = run_once(prices, track, now)
    json.dump(out, open(LATEST_FILE, "w"), indent=2)
    json.dump(track, open(TRACK_FILE, "w"), separators=(",", ":"))
    print(json.dumps(out, indent=2))


def selftest():
    rng = np.random.default_rng(3)
    n = 6000
    idx = pd.date_range("2026-03-02", periods=n, freq="5min", tz="UTC")
    vol = 0.0006 * (1 + 0.6 * np.sin(np.arange(n) / 400) + 0.4 * (idx.hour >= 13))
    full = {a: pd.Series(100 * np.exp(np.cumsum(rng.standard_t(5, n) * vol)), index=idx) for a in ASSETS}
    track = []
    for end in range(1000, n, 3):                              # a run every 15 minutes
        now = idx[end] + timedelta(minutes=6)
        pb = {a: s.iloc[:end + 1] for a, s in full.items()}
        out, track = run_once(pb, track, now.to_pydatetime())
    for a in ASSETS:
        print(a, {f"{m}min": live_coverage(track, a, m) for m in HORIZONS_MIN})
    print("entries:", len(track), "| sample:", out["assets"]["gold"]["ranges"]["10min"])


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
