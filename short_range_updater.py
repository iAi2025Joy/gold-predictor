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
TARGET90 = 0.90
NOMINAL90 = {"gold": 0.90, "oil": 0.92}     # oil calibrated slightly higher, same reason as the 70% band
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


# EVENT WIDENING (from research_range_events: at sudden spikes and at the 8:30 / 9:30 / 10:00 New York data/open bars the
# 70% ranges only held ~59-66% of the time and needed to be ~1.2-1.3x wider; 90% bands held ~82-87%).
EVENT_WIDEN = 1.25


def event_moment(price):
    """'spike', 'data_or_open' or None for the NEWEST bar (uses only data up to that bar)."""
    if len(price) < 150:
        return None
    lp = np.log(price.values)
    r1 = np.diff(lp, prepend=np.nan)
    sd = pd.Series(r1).rolling(288, min_periods=100).std().shift(1).values[-1]
    et = price.index[-1].tz_convert("America/New_York")
    if (et.hour, et.minute) in ((8, 30), (9, 30), (10, 0)):
        return "data_or_open"
    if np.isfinite(sd) and sd > 0:
        z1 = abs(r1[-1]) / sd
        z3 = abs(lp[-1] - lp[-4]) / (sd * np.sqrt(3))
        if z1 > 2.5 or z3 > 2.5:
            return "spike"
    return None


EVENT_FILE = "event_calendar.json"
CALENDAR_WIDEN = {"high": 1.25, "medium": 1.10}   # high factor = same as measured at 8:30/9:30/10:00 NY; medium is a cautious half-step
CALENDAR_MAX_AGE_HOURS = 24 * 9


def load_events(now):
    """Scheduled economic events [(utc datetime, 'high'|'medium', title)] from event_calendar.json; [] if missing/too old."""
    try:
        d = json.load(open(EVENT_FILE))
        upd = datetime.fromisoformat(d["updated_at"].replace("Z", "+00:00"))
        if (now - upd).total_seconds() > CALENDAR_MAX_AGE_HOURS * 3600:
            return []
        return [(datetime.fromisoformat(e["time_utc"].replace("Z", "+00:00")), e["impact"], e["title"]) for e in d.get("events", [])]
    except Exception:
        return []


def calendar_factor(price, H, events):
    """Widening factor + label for scheduled events that fall inside the span this range covers
    (from 10 minutes before the base price's time to the target time)."""
    if not events:
        return 1.0, None
    start = price.index[-1].to_pydatetime() - timedelta(minutes=10)
    end = price.index[-1].to_pydatetime() + timedelta(minutes=BAR_MIN * (H + 1))
    best, label = 1.0, None
    for t, impact, title in events:
        if start <= t <= end and CALENDAR_WIDEN.get(impact, 1.0) > best:
            best, label = CALENDAR_WIDEN[impact], f"calendar: {title} ({impact})"
    return best, label


def make_range(price, H, asset, event_widen=True, events=None, nominal_adj=(0.0, 0.0)):
    """Calibrated range for H bars ahead from the last bar of `price`. None if not enough data."""
    lp = np.log(price.values)
    n = len(lp)
    if n < CAL_POINTS + VOL_BARS + H + 5:
        return None
    r1 = np.diff(lp, prepend=np.nan)
    # VOLATILITY ESTIMATE: exponentially weighted (alpha 0.03 per 5-minute bar, about a 5-hour memory) instead of a flat 36-bar window.
    # Walk-forward test (research_range_sharpness, ~60 days of real bars): tighter ranges at the same coverage in all 8
    # asset x horizon combinations (interval score 0.3% to 2.5% better). Small but consistent; it costs nothing.
    vol = np.sqrt(pd.Series(r1 ** 2).ewm(alpha=0.03, adjust=False).mean().values)
    vol[:VOL_BARS] = np.nan
    t = n - 1
    if not np.isfinite(vol[t]) or vol[t] <= 0:
        return None
    js = np.arange(t - CAL_POINTS - H, t - H + 1)                    # moves fully known at time t
    js = js[js >= 0]
    rH = lp[js + H] - lp[js]
    zall = rH / (vol[js] * np.sqrt(H))
    ok = np.isfinite(zall)
    z, rsel = zall[ok], rH[ok]
    method = "global"
    if SESSION_AWARE[asset]:
        sess = np.array([session(price.index[j]) for j in js])
        m2 = ok & (sess == session(price.index[t]))
        if int(m2.sum()) >= MIN_Z:
            z, rsel, method = zall[m2], rH[m2], "same_session"
    if len(z) < MIN_Z:
        return None
    nom = min(0.95, max(0.50, NOMINAL[asset] + nominal_adj[0]))
    z_lo, z_hi = np.quantile(z, [(1 - nom) / 2, 1 - (1 - nom) / 2])
    n90 = min(0.99, max(0.70, NOMINAL90[asset] + nominal_adj[1]))
    z_lo9, z_hi9 = np.quantile(z, [(1 - n90) / 2, 1 - (1 - n90) / 2])
    base = float(price.iloc[-1]); s = vol[t] * np.sqrt(H)
    moment = event_moment(price) if event_widen else None
    f = EVENT_WIDEN if moment else 1.0
    if event_widen and events:
        cf, clabel = calendar_factor(price, H, events)
        if cf > f:
            f, moment = cf, clabel
    return {"range_low": round(base * float(np.exp(f * z_lo * s)), 4), "range_high": round(base * float(np.exp(f * z_hi * s)), 4),
            "range_low90": round(base * float(np.exp(f * z_lo9 * s)), 4), "range_high90": round(base * float(np.exp(f * z_hi9 * s)), 4),
            "base_price": base, "method": method, "n_calibration": int(len(z)),
            "widened_for": moment,
            # share of recent comparable windows in which price ended HIGHER (descriptive only; studies found no predictive lean)
            "chance_higher": round(float(np.mean(rsel > 0)), 3)}


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
        if e.get("range_low90") is not None:
            e["hit90"] = bool(e["range_low90"] <= actual <= e["range_high90"])
    return track


# ---------------------------------------------------------------- SELF-CALIBRATION
# The system learns from its own scored ranges: if a band keeps landing OUTSIDE its target hit rate over the last 300 scored
# ranges, the requested coverage is nudged (half the error, at most once per 6 hours, never more than +/-6 points, and only after
# 100 scored ranges and an error bigger than the 'dead band'). Ranges already widened for events are left out of the measurement.
CAL_FILE = "calibration_state.json"
CAL_MIN_N, CAL_WINDOW = 100, 300
CAL_DEADBAND = {"hit": 0.04, "hit90": 0.03}
CAL_GAIN, CAL_MAX, CAL_EVERY_HOURS = 0.5, 0.06, 6


def load_calibration():
    try:
        return json.load(open(CAL_FILE))
    except Exception:
        return {"adj": {}, "last_update": {}, "log": []}


def save_calibration(cal):
    json.dump(cal, open(CAL_FILE, "w"), indent=1)


def get_adj(cal, asset, m):
    a = cal.get("adj", {}).get(asset, {}).get(f"{m}min", {})
    return float(a.get("adj70", 0.0)), float(a.get("adj90", 0.0))


def update_calibration(cal, track, now):
    changed = []
    for asset in ASSETS:
        for m in HORIZONS_MIN:
            for key, target, field in (("hit", TARGET, "adj70"), ("hit90", TARGET90, "adj90")):
                lk = f"{asset}|{m}|{key}"
                last = cal["last_update"].get(lk)
                if last and (now - datetime.fromisoformat(last)).total_seconds() < CAL_EVERY_HOURS * 3600:
                    continue
                h = [e[key] for e in track if e["asset"] == asset and e["horizon_min"] == m
                     and e.get(key) is not None and not e.get("widened_for")][-CAL_WINDOW:]
                if len(h) < CAL_MIN_N:
                    continue
                cov = sum(h) / len(h)
                err = target - cov
                if abs(err) <= CAL_DEADBAND[key]:
                    continue
                slot = cal["adj"].setdefault(asset, {}).setdefault(f"{m}min", {"adj70": 0.0, "adj90": 0.0})
                old = slot[field]
                new = max(-CAL_MAX, min(CAL_MAX, old + CAL_GAIN * err))
                if abs(new - old) < 1e-9:
                    continue
                slot[field] = round(new, 4)
                cal["last_update"][lk] = now.isoformat()
                changed.append({"time": now.isoformat(timespec="minutes"), "asset": asset, "horizon": f"{m}min", "band": field,
                                "measured_hit_rate": round(cov, 3), "n": len(h), "adjustment_before": old, "adjustment_after": slot[field]})
    cal["log"] = (cal.get("log", []) + changed)[-50:]
    return changed


def live_coverage(track, asset, horizon, key="hit", target=TARGET):
    h = [e[key] for e in track if e["asset"] == asset and e["horizon_min"] == horizon and e.get(key) is not None][-LIVE_WINDOW:]
    n = len(h)
    if n == 0:
        return {"live_coverage": None, "live_n": 0, "live_note": "No scored live ranges yet."}
    k = int(sum(h)); c = k / n
    if n < 30:
        note = f"{k}/{n} scored so far -- too few to judge."
    else:
        p = stats.binomtest(k, n, target).pvalue
        note = f"{c:.1%} of the last {n} ranges contained the real price (target {target:.0%}). " + \
               ("Consistent with target." if p >= 0.01 else "Meaningfully OFF target -- treat with caution.")
    return {"live_coverage": round(c, 4), "live_n": n, "live_note": note}


def one_sided(track, asset, horizon):
    """How often the real price ended ABOVE the low end / BELOW the high end of the 70% range (last LIVE_WINDOW scored checks).
    A central 70% range should leave about 15% in each tail, so each side should hold about 85% of the time."""
    h = [e for e in track if e["asset"] == asset and e["horizon_min"] == horizon and e.get("actual_price") is not None
         and e.get("hit") is not None][-LIVE_WINDOW:]
    n = len(h)
    if n == 0:
        return {"stay_above_low_rate": None, "stay_below_high_rate": None, "one_sided_n": 0}
    return {"stay_above_low_rate": round(sum(1 for e in h if e["actual_price"] >= e["range_low"]) / n, 4),
            "stay_below_high_rate": round(sum(1 for e in h if e["actual_price"] <= e["range_high"]) / n, 4), "one_sided_n": n}


def run_once(price_by_asset, track, now, cal=None):
    track = score_pending(track, price_by_asset, now)
    cal = cal if cal is not None else {"adj": {}, "last_update": {}, "log": []}
    update_calibration(cal, track, now)
    existing = {(e["asset"], e["horizon_min"], e["base_time"]) for e in track}
    out = {"updated_at": now.isoformat(), "coverage_target": TARGET, "assets": {},
           "method_note": ("Calibrated price ranges, NOT direction forecasts: direction cannot be predicted better than chance "
                           "(strict walk-forward studies). Each range says where the price is likely to be at the horizon; "
                           "live_coverage is how often real prices actually landed inside these ranges."),
           "data_note": "Based on Yahoo Finance futures 5-minute bars (GC=F, CL=F), which can be delayed several minutes; "
                        "because the newest price is already several minutes old, each range covers the delay PLUS the horizon "
                        "(see spans_minutes_of_movement) so that it is honest about 'minutes from now'."}
    events = load_events(now)
    for asset, price in price_by_asset.items():
        if price is None or not len(price):
            out["assets"][asset] = {"status": "no_data"}; continue
        base_time = price.index[-1]
        t0 = base_time + timedelta(minutes=BAR_MIN)                      # the base bar's CLOSE time = when that price was real
        age_min = (now - t0).total_seconds() / 60.0                      # how old the newest price is (Yahoo futures are delayed ~10-15 min)
        entry = {"status": "ok", "data_as_of": t0.isoformat(), "base_price": float(price.iloc[-1]),
                 "data_age_minutes": round(age_min, 1), "ranges": {}}
        if age_min > 40:
            entry["status"] = "stale"
            entry["note"] = "Market data is more than 40 minutes old (market closed or feed delayed); ranges not issued."
            out["assets"][asset] = entry; continue
        for m in HORIZONS_MIN:
            # The price we hold is already `age_min` old, so a range for "m minutes from NOW" must span
            # age + m minutes of price movement. We round up to whole 5-minute bars (never narrower than needed).
            k = max(1, int(np.ceil((age_min + m) / BAR_MIN)))
            if k > 24:
                entry["ranges"][f"{m}min"] = {"status": "data_too_old"}; continue
            rg = make_range(price, k, asset, events=events, nominal_adj=get_adj(cal, asset, m))
            if rg is None:
                entry["ranges"][f"{m}min"] = {"status": "insufficient_data"}; continue
            tt = base_time + timedelta(minutes=k * BAR_MIN)               # bar whose close is the scoring target
            key = (asset, m, base_time.isoformat())
            if key not in existing:
                track.append({"asset": asset, "horizon_min": m, "base_time": base_time.isoformat(),
                              "target_time": tt.isoformat(), "base_price": rg["base_price"],
                              "range_low": rg["range_low"], "range_high": rg["range_high"],
                              "range_low90": rg["range_low90"], "range_high90": rg["range_high90"],
                              "method": rg["method"], "widened_for": rg.get("widened_for"), "hit": None, "hit90": None})
            l90 = live_coverage(track, asset, m, "hit90", TARGET90)
            entry["ranges"][f"{m}min"] = {"range_low": rg["range_low"], "range_high": rg["range_high"],
                                          "range_low90": rg["range_low90"], "range_high90": rg["range_high90"],
                                          "live_coverage90": l90["live_coverage"], "live_n90": l90["live_n"], "live_note90": l90["live_note"],
                                          "horizon_from_now_min": m, "spans_minutes_of_movement": k * BAR_MIN,
                                          "chance_higher": rg["chance_higher"], "chance_lower": round(1 - rg["chance_higher"], 3),
                                          "valid_at": (t0 + timedelta(minutes=k * BAR_MIN)).isoformat(), "widened_for": rg.get("widened_for"),
                                          "self_calibration": {"adj70": get_adj(cal, asset, m)[0], "adj90": get_adj(cal, asset, m)[1]},
                                          "method": rg["method"], **live_coverage(track, asset, m), **one_sided(track, asset, m)}
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
    cal = load_calibration()
    out, track = run_once(prices, track, now, cal=cal)
    save_calibration(cal)
    out["self_calibration_log"] = cal.get("log", [])[-5:]
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
        now = idx[end] + timedelta(minutes=15)       # emulate Yahoo's ~10 min delay + bar length
        pb = {a: s.iloc[:end + 1] for a, s in full.items()}
        out, track = run_once(pb, track, now.to_pydatetime())
    for a in ASSETS:
        print(a, {f"{m}min": live_coverage(track, a, m) for m in HORIZONS_MIN})
    print("entries:", len(track), "| sample:", out["assets"]["gold"]["ranges"]["10min"])


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
