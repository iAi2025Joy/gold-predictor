"""
macro_data_fetcher.py
========================

Fetches three macro series from FRED (St. Louis Fed) that are gold's most
direct, well-established price drivers -- the ones a short-term trader
would actually watch instead of staring at gold's own chart:

  - DFII10  : 10-Year Treasury Inflation-Indexed Security, Constant
              Maturity -- this IS "real yields" (nominal yield minus
              inflation expectations). Real yields are the opportunity
              cost of holding non-yielding gold; this is arguably gold's
              single most direct macro driver.
  - DTWEXBGS: Trade Weighted U.S. Dollar Index (Broad, Goods & Services)
              -- a free, reliable DXY-style dollar-strength proxy. Gold
              is priced in USD, so dollar moves are close to mechanically
              inverse to gold moves.
  - VIXCLS  : CBOE Volatility Index (VIX) close -- risk-sentiment proxy.
              Rising VIX often (not always) coincides with safe-haven
              demand for gold.

HONEST LIMITATION, stated plainly (same standard as the rest of this
project): all three are DAILY series on FRED, not intraday. VIXCLS in
particular is typically only published the next business day morning.
Running this more than a few times a day will just re-fetch the same
value until FRED updates it -- that's expected, not a bug. Scheduled at
the same 6-hour cadence as news_sentiment_fetcher.py for that reason (see
update_macro_data.yml).

SETUP REQUIRED
-----------------
Free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html,
set as the FRED_API_KEY repository secret.
"""

import os
import json
from datetime import datetime, timezone

import requests

API_KEY = os.environ.get("FRED_API_KEY", "")
OUTPUT_FILE = "macro_history.json"

SERIES = {
    "real_yield_10y": "DFII10",
    "dxy": "DTWEXBGS",
    "vix": "VIXCLS",
}


def fetch_latest_value(series_id):
    """Fetch the most recent real observation for a FRED series.

    FRED returns "." (a literal dot string) for non-trading days/holidays
    instead of omitting the row -- so we walk back through the most recent
    observations and take the first one that is an actual number, rather
    than trusting the single latest row blindly."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 10,  # a few days of buffer in case of holidays/missing rows
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    for obs in data.get("observations", []):
        if obs.get("value") not in (None, ".", ""):
            return float(obs["value"]), obs["date"]
    return None, None


def fetch_macro_snapshot():
    if not API_KEY:
        raise RuntimeError("FRED_API_KEY is not set.")

    result = {"timestamp": datetime.now(timezone.utc).isoformat()}
    any_success = False

    for feature_name, series_id in SERIES.items():
        try:
            value, obs_date = fetch_latest_value(series_id)
            result[feature_name] = value
            result[f"{feature_name}_as_of"] = obs_date
            if value is not None:
                any_success = True
        except Exception as err:
            # One series failing (FRED hiccup, series temporarily
            # unavailable) shouldn't take down the other two -- record it
            # honestly as missing rather than aborting the whole snapshot.
            print(f"Could not fetch {series_id} ({feature_name}): {err}")
            result[feature_name] = None
            result[f"{feature_name}_as_of"] = None

    result["macro_data_available"] = any_success
    return result


def load_history():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_history(history):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(history, f, indent=2)


def main():
    snapshot = fetch_macro_snapshot()

    print("Macro snapshot fetched:")
    print(json.dumps(snapshot, indent=2))

    history = load_history()
    history.append(snapshot)
    save_history(history)
    print(f"\nSaved to {OUTPUT_FILE} ({len(history)} total entries).")


if __name__ == "__main__":
    main()
