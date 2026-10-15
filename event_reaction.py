"""
event_reaction.py -- records how gold reacts to scheduled releases (CPI, PPI, jobs, retail sales, Fed...) and tests, honestly,
whether the SURPRISE (actual result vs forecast) predicts the direction of the move afterwards.

Why: direction from prices alone has no measured edge at 10-60 minutes. A release surprise is NEW information (not in earlier
prices), and the usual market reasoning is simple: hotter inflation / stronger data -> higher yields and dollar -> gold down.

What it does every cycle (called from live_tick_recorder.py, no extra schedule):
  1. Looks at event_calendar.json for releases that happened in the last 3 hours.
  2. Saves one record per release moment: titles, forecast, previous, ACTUAL (if a source supplies it), the gold price at the release
     and 10 / 20 / 30 / 60 minutes later (from live_gold_ticks.json, our own 2-minute ticks).
  3. Where actual and forecast are both known, computes the surprise and, using a rule FIXED IN ADVANCE (below, not fitted to results),
     the expected gold direction.
  4. analyse(): scores that rule against what gold really did, with a binomial test vs a coin flip and vs the commoner direction.
     Nothing is validated before 20 releases with a surprise at that horizon, and a Bonferroni correction over 4 horizons applies.
GARNET only uses it if analyse() says validated. Until then it is just a record.

ACTUAL values: the free weekly calendar normally has only forecast/previous. This module also tries Finnhub's economic calendar
(needs FINNHUB_API_KEY; it may be a paid feature -- if refused, "actual_source" says so and releases are still logged with the gold reaction).
Files: events_reaction.json.   Run: python event_reaction.py     Self-test: python event_reaction.py selftest
"""
import sys, json, os, re
from datetime import datetime, timezone, timedelta

CAL_FILE, TICK_FILE, OUT = "event_calendar.json", "live_gold_ticks.json", "events_reaction.json"
HORIZONS = (10, 20, 30, 60)
WINDOW_H = 3
MIN_EVENTS = 20

# Direction rule FIXED IN ADVANCE: expected gold move = SIGN * sign(actual - forecast).
# Higher inflation / stronger activity / more jobs -> tighter policy expected -> gold DOWN (-1).
# Higher unemployment / jobless claims -> weaker economy -> gold UP (+1). Anything not listed is logged but not scored.
FAMILIES = [
    (r"\bcpi\b|\bppi\b|\bpce\b|price index|inflation", "inflation", -1),
    (r"non-?farm|employment change|\badp\b|average hourly earnings|payrolls", "jobs_strength", -1),
    (r"unemployment|jobless claims", "jobs_weakness", +1),
    (r"retail sales|\bgdp\b|\bism\b|pmi|durable goods", "activity", -1),
]

def family(title):
    t = title.lower()
    for pat, name, sign in FAMILIES:
        if re.search(pat, t):
            return name, sign
    return None, 0

def parse_num(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([%KMBT]?)", s, re.I)
    if not m:
        return None
    v = float(m.group(1)); u = m.group(2).upper()
    return v * {"": 1, "%": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[u]

def tick_at(ticks, t, tol_min=4):
    best, bd = None, None
    for k in ticks:
        d = abs((k["_t"] - t).total_seconds()) / 60
        if d <= tol_min and (bd is None or d < bd):
            best, bd = k, d
    return best["p"] if best else None

def load_ticks():
    try:
        raw = json.load(open(TICK_FILE)); raw = raw.get("ticks", raw)
        return [{"_t": datetime.fromisoformat(x["t"]), "p": float(x["p"])} for x in raw]
    except Exception:
        return []

def fetch_actuals(now):
    """Returns ({(title, time_utc): actual_string}, status_text). Never raises."""
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        return {}, "no FINNHUB_API_KEY in this job"
    try:
        import requests
        d0 = (now - timedelta(days=1)).strftime("%Y-%m-%d"); d1 = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        r = requests.get("https://finnhub.io/api/v1/calendar/economic", params={"from": d0, "to": d1, "token": key}, timeout=20)
        if r.status_code != 200:
            return {}, f"Finnhub economic calendar refused (HTTP {r.status_code}; likely a paid feature)"
        out = {}
        for e in (r.json().get("economicCalendar") or []):
            if e.get("country") == "US" and e.get("actual") is not None:
                out[(str(e.get("event", "")).lower(), str(e.get("time", ""))[:16])] = (e.get("actual"), e.get("estimate"), e.get("unit"))
        return out, f"Finnhub ok ({len(out)} US actuals)"
    except Exception as ex:
        return {}, f"Finnhub error: {ex}"

def update(now=None, cal=None, ticks=None, actuals=None, status=None, records=None):
    now = now or datetime.now(timezone.utc)
    if cal is None:
        try: cal = json.load(open(CAL_FILE))["events"]
        except Exception: cal = []
    ticks = load_ticks() if ticks is None else ticks
    if records is None:
        try: records = json.load(open(OUT))["releases"]
        except Exception: records = []
    if actuals is None:
        actuals, status = fetch_actuals(now)
    by_key = {r["time_utc"]: r for r in records}
    groups = {}
    for e in cal:
        t = datetime.fromisoformat(e["time_utc"])
        if timedelta(0) <= now - t <= timedelta(hours=WINDOW_H) or e["time_utc"] in by_key:
            groups.setdefault(e["time_utc"], []).append(e)
    for tu, evs in groups.items():
        t = datetime.fromisoformat(tu)
        rec = by_key.get(tu) or {"time_utc": tu, "events": [], "gold": {}}
        if tu not in by_key:
            records.append(rec); by_key[tu] = rec
        rec["events"] = []
        for e in evs:
            fam, sign = family(e["title"])
            act = None
            for (name, tm), val in actuals.items():
                if name == e["title"].lower() and tm == tu[:16].replace("T", " "):
                    act = val[0]
            if act is None:
                act = e.get("actual")
            row = {"title": e["title"], "family": fam, "sign": sign, "forecast": e.get("forecast"), "previous": e.get("previous"), "actual": act}
            a, f = parse_num(act), parse_num(e.get("forecast"))
            if a is not None and f is not None:
                row["surprise"] = round(a - f, 6)
            rec["events"].append(row)
        g = rec["gold"]
        if "p0" not in g:
            p0 = tick_at(ticks, t)
            if p0 is not None: g["p0"] = p0
        if "p0" in g:
            for h in HORIZONS:
                if f"p{h}" not in g and now >= t + timedelta(minutes=h):
                    ph = tick_at(ticks, t + timedelta(minutes=h))
                    if ph is not None: g[f"p{h}"] = ph
    records.sort(key=lambda r: r["time_utc"])
    return records[-400:], status

def expected_direction(rec):
    """Combined expectation for one release moment: sum of sign*sign(surprise) over scoreable events; None if no surprise known or a tie."""
    s = 0
    for e in rec["events"]:
        if e.get("surprise") not in (None, 0) and e.get("sign"):
            s += e["sign"] * (1 if e["surprise"] > 0 else -1)
    return None if s == 0 else ("up" if s > 0 else "down")

def analyse(records):
    from scipy import stats
    out = {"min_events": MIN_EVENTS, "horizons": {}, "n_releases_logged": len(records),
           "n_with_surprise": sum(1 for r in records if expected_direction(r)), "validated": []}
    for h in HORIZONS:
        n = k = ups = 0
        for r in records:
            d, g = expected_direction(r), r["gold"]
            if d and "p0" in g and f"p{h}" in g and g[f"p{h}"] != g["p0"]:
                real = "up" if g[f"p{h}"] > g["p0"] else "down"
                n += 1; k += (real == d); ups += (real == "up")
        row = {"n": n}
        if n:
            base = max(ups / n, 1 - ups / n)
            p = stats.binomtest(k, n, max(0.5, base), alternative="greater").pvalue
            row.update({"correct": k, "accuracy": round(k / n, 3), "baseline": round(base, 3), "p_corrected": round(float(min(1, p * len(HORIZONS))), 4)})
            if n >= MIN_EVENTS and k / n > max(0.5, base) and row["p_corrected"] < 0.05:
                out["validated"].append(f"{h}min")
        out["horizons"][f"{h}min"] = row
    out["verdict"] = (f"Release-surprise rule validated at: {', '.join(out['validated'])}" if out["validated"] else
                      f"Not validated yet: {out['n_with_surprise']} releases with a known surprise (needs at least {MIN_EVENTS}).")
    return out

def run_once():
    recs, status = update()
    summary = analyse(recs)
    json.dump({"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "actual_source": status,
               "rule": "gold direction = sign(actual - forecast) x fixed family sign (inflation/strength: down; unemployment/claims: up)",
               "analysis": summary, "releases": recs}, open(OUT, "w"), indent=1)
    print(summary["verdict"], "| actual source:", status)

# ---------------------------------------------------------------- selftest
def selftest():
    import random
    random.seed(4)
    def world(follow):
        cal, ticks, now0 = [], [], datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
        t = now0; p = 2000.0
        while t < now0 + timedelta(days=60):
            ticks.append({"_t": t, "p": p}); t += timedelta(minutes=2); p += random.gauss(0, 0.4)
        recs = []
        for i in range(40):
            tr = now0 + timedelta(days=1 + i, hours=12, minutes=30)
            sur = random.choice([-0.2, 0.2]); d = -1 if sur > 0 else 1            # CPI: hotter -> gold down
            move = (d if random.random() < follow else -d) * random.uniform(2, 6)
            for k in ticks:                                                     # inject a drift after the release
                dm = (k["_t"] - tr).total_seconds() / 60
                if 0 <= dm <= 60: k["p"] += move * dm / 60
                elif dm > 60 and dm < 600: k["p"] += move
            cal.append({"title": "CPI m/m", "country": "USD", "time_utc": tr.isoformat(timespec="minutes"), "impact": "high", "forecast": "0.3%", "previous": "0.3%"})
            recs.append((tr, f"{0.3 + sur:.1f}%"))
        return cal, ticks, recs, now0
    ok = True
    for name, follow in (("null (50%)", 0.5), ("real effect (80%)", 0.8)):
        cal, ticks, rr, now0 = world(follow)
        actuals = {("cpi m/m", tr.strftime("%Y-%m-%d %H:%M")): (a, "0.3%", "%") for tr, a in rr}
        recs = []
        for tr, _ in rr:
            recs, _s = update(now=tr + timedelta(minutes=90), cal=cal, ticks=ticks, actuals=actuals, status="selftest", records=recs)
        res = analyse(recs)
        print(f"SELFTEST {name}: releases={res['n_releases_logged']} with_surprise={res['n_with_surprise']} 30min={res['horizons']['30min']} -> {res['verdict']}")
        if follow == 0.5 and res["validated"]: ok = False; print("FAIL: null case must not validate")
        if follow == 0.8 and not res["validated"]: ok = False; print("FAIL: real effect should validate")
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    selftest() if len(sys.argv) > 1 and sys.argv[1] == "selftest" else run_once()
