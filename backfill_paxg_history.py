"""
backfill_paxg_history.py
===========================

ONE-TIME script (not part of the recurring 8-hour workflow) to seed
gold_price_history.json with a real year of properly-matched historical
data, so the prediction model doesn't need to wait ~20 days to accumulate
enough live points from scratch.

WHY PAXG, NOT THE OLD CSV
----------------------------
The earlier seed (a 2008-2018 GLD ETF share-price CSV) caused a real bug:
it was in different units than GoldAPI's live spot-price feed, producing
a nonsensical prediction. This script avoids that two ways:

  1. Units: PAX Gold (PAXG) is a cryptocurrency token backed 1:1 by
     physical gold (1 token = 1 fine troy ounce), traded on Binance as
     PAXGUSDT. Its USD price tracks spot gold closely -- confirmed
     directly: at the time of writing, PAXG traded around $4,000-4,130,
     matching GoldAPI's own live quote (~$4,127) to within normal
     market variation. It is NOT a perfect 1:1 match to LBMA spot gold
     at every instant (PAXG carries a small, variable exchange
     premium/discount), so treat this as a very close, disclosed
     approximation, not an official identical price.

  2. Resolution: Binance's public kline API supports an exact "8h"
     interval, matching this project's own update cadence exactly --
     no resolution mismatch with the live feed going forward either.

This is Binance's PUBLIC market-data endpoint: no API key or account
needed, confirmed directly against a real request (see the project's
build notes).

HOW TO USE
-------------
Run this ONCE, locally or via a manual GitHub Actions dispatch, to
produce gold_price_history.json. Commit that file into the gold-predictor
repo. From then on, the normal recurring workflow (gold_predictor_updater.py)
will load this file and keep appending real GoldAPI live points to it,
exactly as it already does.
"""

import json
import time
from datetime import datetime, timezone

import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "PAXGUSDT"
INTERVAL = "8h"
DAYS_BACK = 365
OUTPUT_FILE = "gold_price_history.json"


def fetch_klines_page(start_time_ms, limit=1000):
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_time_ms,
        "limit": limit,
    }
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def backfill():
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - DAYS_BACK * 24 * 60 * 60 * 1000

    all_records = []
    cursor = start_ms
    page = 0
    max_pages = 5  # safety cap; a real year of 8h data needs ~2 pages, this guards against any unexpected API behavior looping indefinitely
    while page < max_pages:
        page += 1
        klines = fetch_klines_page(cursor)
        if not klines:
            print(f"Page {page}: no data returned, stopping (reached end of available data).")
            break
        for k in klines:
            open_time_ms, open_p, high_p, low_p, close_p = k[0], k[1], k[2], k[3], k[4]
            dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
            all_records.append({"Date": dt.strftime("%Y-%m-%d %H:%M:%S"), "GLD": float(close_p)})
        print(f"Page {page}: fetched {len(klines)} candles "
              f"({all_records[0]['Date']} to {all_records[-1]['Date']} so far)")
        last_open_time = klines[-1][0]
        if len(klines) < 1000 or last_open_time >= now_ms:
            # either we got a partial page (no more data available), or we've
            # caught up to the actual present moment -- either way, stop
            print(f"Reached end of available data after page {page}.")
            break
        cursor = last_open_time + 1
        time.sleep(0.5)  # be polite to the public endpoint, no rate-limit issues expected at this volume
    else:
        print(f"WARNING: hit the {max_pages}-page safety cap without reaching the present. "
              f"This shouldn't happen for a 1-year/8h backfill -- check the output before trusting it.")

    # de-duplicate by date, just in case of any pagination overlap
    seen = set()
    deduped = []
    for r in all_records:
        if r["Date"] not in seen:
            seen.add(r["Date"])
            deduped.append(r)
    deduped.sort(key=lambda r: r["Date"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(deduped, f)

    print(f"\nWrote {len(deduped)} records to {OUTPUT_FILE}")
    print(f"Range: {deduped[0]['Date']} to {deduped[-1]['Date']}")
    print(f"Price range: ${min(r['GLD'] for r in deduped):.2f} to ${max(r['GLD'] for r in deduped):.2f}")


if __name__ == "__main__":
    backfill()
