"""
event_calendar_fetcher.py -- saves the week's scheduled economic events (central banks, US jobs / inflation data,
oil inventories, ...) to event_calendar.json, so the price ranges can be widened around them.

Source: the free public weekly calendar JSON from Forex Factory's data partner (this week + next week).
Kept: USD events of High or Medium impact, plus High-impact EUR / GBP / JPY / CNY events (they also move gold and oil).
If the download fails the OLD file is left untouched (stale data is better than none; the watchdog warns if it gets old).

Run: python event_calendar_fetcher.py        Self-test: python event_calendar_fetcher.py selftest
"""
import sys, json
from datetime import datetime, timezone

URLS = ["https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://nfs.faireconomy.media/ff_calendar_nextweek.json"]
OUT = "event_calendar.json"
KEEP_HIGH_FROM = {"EUR", "GBP", "JPY", "CNY"}

def normalise(raw):
    """raw = list of calendar rows -> sorted list of {title, country, time_utc, impact, forecast, previous}."""
    out = []
    for r in raw or []:
        try:
            impact = str(r.get("impact", "")).strip().capitalize()
            country = str(r.get("country", "")).strip().upper()
            if impact not in ("High", "Medium"):
                continue
            if not (country == "USD" or (country in KEEP_HIGH_FROM and impact == "High")):
                continue
            t = datetime.fromisoformat(str(r["date"]).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            out.append({"title": str(r.get("title", "")).strip(), "country": country,
                        "time_utc": t.astimezone(timezone.utc).isoformat(timespec="minutes"),
                        "impact": impact.lower(), "forecast": r.get("forecast") or None, "previous": r.get("previous") or None})
        except Exception:
            continue
    out.sort(key=lambda e: e["time_utc"])
    return out

def fetch():
    import requests
    rows = []
    for u in URLS:
        try:
            r = requests.get(u, timeout=20, headers={"User-Agent": "Mozilla/5.0 (GARNET calendar fetcher)"})
            if r.status_code == 200:
                rows += r.json()
            else:
                print("calendar", u, "HTTP", r.status_code)
        except Exception as e:
            print("calendar fetch failed", u, e)
    return rows

def main():
    rows = fetch()
    events = normalise(rows)
    if not rows:
        print("No calendar data downloaded -- leaving the existing file unchanged."); sys.exit(0)
    json.dump({"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "source": "faireconomy weekly calendar (this + next week)",
               "note": "Scheduled events (times in UTC). Used to widen price ranges around releases; not a prediction.",
               "events": events}, open(OUT, "w"), indent=1)
    print(f"saved {len(events)} events (from {len(rows)} calendar rows)")
    for e in events[:12]: print("  ", e["time_utc"], e["country"], e["impact"], e["title"])

def selftest():
    sample = [
        {"title": "CPI m/m", "country": "USD", "date": "2026-10-14T08:30:00-04:00", "impact": "High", "forecast": "0.3%", "previous": "0.2%"},
        {"title": "Crude Oil Inventories", "country": "USD", "date": "2026-10-14T10:30:00-04:00", "impact": "Medium", "forecast": "", "previous": "1.2M"},
        {"title": "Bank Holiday", "country": "USD", "date": "2026-10-14T00:00:00-04:00", "impact": "Holiday"},
        {"title": "German ZEW", "country": "EUR", "date": "2026-10-14T05:00:00-04:00", "impact": "Medium"},
        {"title": "ECB Rate Decision", "country": "EUR", "date": "2026-10-15T08:15:00-04:00", "impact": "High"},
        {"title": "Junk row"},
    ]
    ev = normalise(sample)
    assert [e["title"] for e in ev] == ["CPI m/m", "Crude Oil Inventories", "ECB Rate Decision"], ev
    assert ev[0]["time_utc"] == "2026-10-14T12:30+00:00", ev[0]
    print("SELFTEST PASSED"); print(json.dumps(ev[0]))

if __name__ == "__main__":
    selftest() if (len(sys.argv) > 1 and sys.argv[1] == "selftest") else main()
