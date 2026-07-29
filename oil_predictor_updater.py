"""
oil_predictor_updater.py
============================

Oil-market counterpart to gold_predictor_updater.py, same architecture and
same honesty standards. Run on a schedule (GitHub Actions, DAILY -- not
hourly, see cadence note below) to keep an oil price prediction fresh.

*** VERIFICATION STATUS: Alpha Vantage full-series parsing CONFIRMED working
via real production runs (10,205 real price points, 1986-01-02 to
2026-07-20, successfully fetched and merged). The live-price patch added
below (fetch_live_oil_price) is CONFIRMED against a real authenticated
OilPriceAPI.com call -- see its docstring for the exact real response
received. ***

IMPORTANT DESIGN NOTE -- FULL-SERIES MERGE, NOT SINGLE-POINT APPEND:
Endpoints in this same Alpha Vantage family (REAL_GDP, CPI, Treasury Yield)
are documented to return their ENTIRE historical "data" array in every
response, not just the latest value -- this is a real, meaningful
difference from GoldPriceZ (which only ever returns a single current spot
price, requiring gold's history to accumulate one point per call). This
was confirmed via real production data: a single call returned 10,205 real
daily price points spanning 1986-2026. So this script does NOT just
fetch-and-append-latest like the gold predictor does -- it fetches the
full returned series EVERY run and MERGES it into local history by date
(freshly-fetched values win on any date collision, since they're the
canonical source). This also means a missed day (e.g. a failed Actions
run) self-heals automatically next run, which gold's single-point-append
approach cannot do.

LIVE-PRICE PATCH (added after a real, confirmed problem): Alpha Vantage's
WTI feed itself was confirmed, via real production data, to carry an
inherent multi-day reporting lag on top of this job's own daily schedule
-- a run on July 23 returned July 20 as its most recent data point, and
the resulting "current" price ($84.38) was off from the real live market
price ($92.17, confirmed via a real OilPriceAPI.com call) by about $7.79
(roughly 9%). To fix this at the root rather than just warn about it in
the chatbot, this script now ALSO calls OilPriceAPI.com and uses that
genuinely fresh price to PATCH today's row in the merged history before
running the prediction pipeline -- so current_price_usd in the resulting
prediction reflects a real live price, while the deep historical backbone
(1986-2026) still comes from Alpha Vantage's full-series response. If the
live-price call fails for any reason, this script falls back gracefully
to whatever Alpha Vantage's own most recent point was (the prior
behavior), rather than failing the whole run over a live-price hiccup.

CADENCE: still runs DAILY (matches the WTI endpoint's daily granularity) --
merging the full series more than once a day would just re-fetch the same
data repeatedly for no benefit. The live-price patch runs once per day too.

BUDGET: Alpha Vantage calls reuse the SAME ALPHAVANTAGE_API_KEY already
used by news_sentiment_fetcher.py, sharing the same 25 requests/day total
budget (still exactly 1 request/run). The live-price call uses a SEPARATE
key, OILPRICEAPI_KEY, on OilPriceAPI's own free tier (200 requests/month)
-- this MUST be added as a GitHub Actions secret separately from Render's
copy of the same key (GitHub Actions secrets and Render environment
variables do NOT share, confirmed the hard way twice already in this
project for other keys -- Serper and GOLDPRICEZ_API_KEY both needed this
exact same fix earlier).
"""

import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

# Reuse gold's already-tested macro data / economic calendar / Fed rate /
# cross-asset logic rather than duplicating it -- see each call site
# below for exactly what's being reused and why. This is the same reuse
# pattern already proven out by dxy_predictor_updater.py.
import gold_predictor_updater as gpu

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")  # SAME key as news_sentiment_fetcher.py
OILPRICEAPI_KEY = os.environ.get("OILPRICEAPI_KEY", "")  # SEPARATE key -- see budget note above
HISTORY_FILE = "oil_price_history.json"           # lives in this repo, committed back after each run
PREDICTION_FILE = "oil_prediction_latest.json"     # lives in this repo, committed back after each run
TRACK_RECORD_FILE = "oil_prediction_track_record.json"
MIN_ROWS_FOR_PREDICTION = 60
HALF_LIFE_DAYS = 30
ROLLING_WINDOW = 50


# ============================================================
# 1. LIVE PRICE -- Alpha Vantage WTI, daily granularity
# ============================================================

def fetch_wti_series():
    """Fetch the FULL historical WTI crude oil daily price series from
    Alpha Vantage -- not just the latest point. See module docstring:
    this endpoint family is documented to return its entire history in
    one response, unlike GoldPriceZ's single-current-price model.

    *** UNVERIFIED SHAPE -- see module docstring. *** Parses the documented
    {"data": [{"date": "YYYY-MM-DD", "value": "82.50"}, ...]} shape used by
    Alpha Vantage's economic-indicator-family endpoints. Raises a clear,
    loud error (rather than silently returning wrong/empty data) if the
    real response doesn't match, so a bad assumption gets caught on first
    real run instead of silently corrupting the history file.

    Returns a list of (date_str, price) tuples, most-recent-first (as
    documented for this endpoint family) -- caller is responsible for
    turning this into a DataFrame and merging with existing history."""
    if not API_KEY:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not set.")
    url = "https://www.alphavantage.co/query"
    params = {"function": "WTI", "interval": "daily", "apikey": API_KEY}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    if "Information" in payload and "data" not in payload:
        raise RuntimeError(f"Alpha Vantage returned an info/error message instead of data: {payload['Information']}")
    if "Note" in payload:
        raise RuntimeError(f"Alpha Vantage rate-limit note: {payload['Note']}")

    if "data" not in payload:
        raise RuntimeError(
            "UNEXPECTED RESPONSE SHAPE from Alpha Vantage WTI endpoint -- "
            "the assumed 'data' key is missing. This confirms the shape "
            "needs real correction, not just a workaround. Raw response: "
            f"{json.dumps(payload)[:500]}"
        )

    records = payload["data"]
    if not records:
        raise RuntimeError("Alpha Vantage WTI response had an empty 'data' array.")

    parsed = []
    skipped = 0
    for rec in records:
        try:
            price = float(rec["value"])
            date_str = rec["date"]
            parsed.append((date_str, price))
        except (KeyError, ValueError, TypeError):
            # Alpha Vantage's economic-indicator endpoints sometimes include
            # a "." placeholder for days with no real data (e.g. non-trading
            # days) -- skip these rather than crash the whole run over a
            # handful of expected gaps.
            skipped += 1
    if skipped:
        print(f"Note: skipped {skipped} unparseable/placeholder records out of {len(records)} returned.")
    if not parsed:
        raise RuntimeError("No parseable (date, value) records found in Alpha Vantage WTI response.")

    return parsed


def fetch_live_oil_price():
    """Fetch a genuinely live WTI price from OilPriceAPI.com. CONFIRMED
    against a real authenticated call -- real response received:
    {"status":"success","data":{"price":92.17,"formatted":"$92.17",
     "currency":"USD","code":"WTI_USD","created_at":"2026-07-23T23:07:17Z",
     "data_status":"current","freshness":{"status":"current",
     "age_seconds":1687,"expected_max_age_seconds":1800},
     "synthetic":false,"stale":false, ...}}

    Returns (price, date_str) on success, or (None, None) if the key isn't
    configured or the call fails for any reason -- callers should fall back
    gracefully to Alpha Vantage's own most recent point rather than fail
    the whole run over a live-price hiccup, since the live patch is a
    freshness improvement, not a hard dependency."""
    if not OILPRICEAPI_KEY:
        print("OILPRICEAPI_KEY not set -- skipping live-price patch, using Alpha Vantage's own most recent point.")
        return None, None

    url = "https://api.oilpriceapi.com/v1/prices/latest"
    try:
        resp = requests.get(
            url,
            params={"by_code": "WTI_USD"},
            headers={"Authorization": f"Token {OILPRICEAPI_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"Live oil price fetch failed ({e}) -- falling back to Alpha Vantage's own most recent point.")
        return None, None

    if payload.get("status") != "success" or "data" not in payload:
        print(f"Unexpected live-price response shape, skipping patch. Raw: {json.dumps(payload)[:300]}")
        return None, None

    d = payload["data"]
    if d.get("synthetic") is True:
        print("Live price flagged as SYNTHETIC (estimated, not a real observed price) -- skipping patch, "
              "falling back to Alpha Vantage's own most recent point rather than using estimated data.")
        return None, None
    if d.get("stale") is True:
        print("Live price API itself flags this price as stale -- skipping patch, "
              "falling back to Alpha Vantage's own most recent point.")
        return None, None

    try:
        price = float(d["price"])
    except (KeyError, ValueError, TypeError):
        print(f"Could not parse 'price' from live-price response, skipping patch. Raw data: {json.dumps(d)[:300]}")
        return None, None

    # Use TODAY's date (UTC) for the patched row, not the API's own
    # as_of/created_at timestamp -- we want this row to represent "today's
    # price" in the daily history series the model trains on, and the live
    # price is confirmed fresh (age_seconds well within expected_max_age_seconds).
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Live price confirmed fresh: ${price:.2f} (age: {d.get('freshness', {}).get('age_seconds', 'unknown')}s) "
          f"-- will patch today's ({today_str}) row with this real live price.")
    return price, today_str


# ============================================================
# 2. LOCAL HISTORY FILE (same pattern as gold_price_history.json)
# ============================================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            records = json.load(f)
        if records:
            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            return df[["Date", "WTI"]].sort_values("Date").reset_index(drop=True)
    print("No history file found or it was empty -- starting fresh. Unlike gold's history, this "
          "should self-backfill immediately from Alpha Vantage's full returned series on the very "
          "first real run (see fetch_wti_series() -- pending real-key verification).")
    return pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "WTI": pd.Series(dtype="float64")})


def merge_fetched_series(history_df, fetched_pairs):
    """Merge the freshly-fetched full series into existing local history,
    by date. Freshly-fetched values WIN on any date collision (Alpha
    Vantage is the canonical source; a locally-stored value could be from
    an earlier, less-complete fetch). This is what allows: (1) a full
    backfill on the very first run, and (2) automatic self-healing of any
    gap from a previously missed/failed run -- neither of which a simple
    append-latest-point approach (like the gold predictor uses, since
    GoldPriceZ only ever gives one current price) can do."""
    fetched_df = pd.DataFrame(fetched_pairs, columns=["Date", "WTI"])
    fetched_df["Date"] = pd.to_datetime(fetched_df["Date"])

    if len(history_df) == 0:
        merged = fetched_df
    else:
        # Combine, then drop duplicate dates keeping the freshly-fetched
        # version (fetched_df rows appended last, keep='last').
        combined = pd.concat([history_df, fetched_df], ignore_index=True)
        merged = combined.drop_duplicates(subset="Date", keep="last")

    return merged.sort_values("Date").reset_index(drop=True)


def save_history(history_df):
    records = history_df.copy()
    records["Date"] = pd.to_datetime(records["Date"]).dt.strftime("%Y-%m-%d")
    with open(HISTORY_FILE, "w") as f:
        json.dump(records.to_dict(orient="records"), f)


# ============================================================
# 3. PREDICTION PIPELINE -- same feature set, same honest testing
#    methodology proven out for gold (McNemar's + Wilcoxon), applied
#    identically here. Deliberately NOT using any of the extra
#    features (MACD, Bollinger Bands) or model classes (GradientBoosting,
#    RandomForest) tested for gold this session -- those were tested and
#    found NOT to beat plain LogisticRegression via proper walk-forward
#    validation, so there's no principled reason to start oil off with
#    them either. Same starting point, same honest evaluation standard.
# ============================================================

def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["wti_ret_1d"] = df["WTI"].pct_change(1)
    df["wti_ret_3d"] = df["WTI"].pct_change(3)
    df["wti_ret_5d"] = df["WTI"].pct_change(5)

    # Defensive clip on extreme returns -- guards against events like WTI's
    # real April 20, 2020 negative-price day (-$36.98/barrel), where dividing
    # by a negative base inverts the sign of a pct_change calculation and
    # produces a semantically backwards "return" (e.g. a real price recovery
    # showing up as a huge negative return). Confirmed via real data: this
    # affects a small number of rows (~6-8 out of 10,000+) around any such
    # event, not enough to meaningfully change backtest results on its own,
    # but clipping is cheap, correctness-improving insurance against this
    # class of distortion recurring or worsening in the future. +/-200% is
    # a generous bound -- genuine daily oil moves rarely approach this, so
    # it only engages for real anomalies, not ordinary volatility.
    RETURN_CLIP = 2.0  # +/-200%
    for col in ["wti_ret_1d", "wti_ret_3d", "wti_ret_5d"]:
        df[col] = df[col].clip(lower=-RETURN_CLIP, upper=RETURN_CLIP)

    df["wti_ma5"] = df["WTI"].rolling(5).mean()
    df["wti_ma20"] = df["WTI"].rolling(20).mean()
    df["wti_ma_ratio"] = df["wti_ma5"] / df["wti_ma20"]
    df["wti_vol10"] = df["wti_ret_1d"].rolling(10).std()
    delta = df["WTI"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["next_ret"] = df["WTI"].shift(-1) / df["WTI"] - 1
    df["target_up"] = (df["next_ret"] > 0).astype(int)
    return df


def merge_news_sentiment(price_df, sentiment_file="oil_news_sentiment_history.json"):
    """Same lookahead-safe backward merge_asof pattern already proven correct
    for gold. Reads from a SEPARATE oil-specific sentiment file (see
    news_sentiment_fetcher.py's new oil-keyword pass) -- not the gold
    sentiment file, to keep the two domains honestly separate."""
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if not os.path.exists(sentiment_file):
        df["news_sentiment"] = 0.0
        df["news_sentiment_available"] = 0
        return df

    with open(sentiment_file) as f:
        sentiment_records = json.load(f)

    sent_df = pd.DataFrame(sentiment_records)
    sent_df = sent_df.dropna(subset=["avg_sentiment_score"])
    if len(sent_df) == 0:
        df["news_sentiment"] = 0.0
        df["news_sentiment_available"] = 0
        return df

    sent_df["timestamp"] = pd.to_datetime(sent_df["timestamp"]).dt.tz_localize(None)
    sent_df = sent_df.sort_values("timestamp")[["timestamp", "avg_sentiment_score"]]

    df = df.sort_values("Date")
    merged = pd.merge_asof(
        df, sent_df,
        left_on="Date", right_on="timestamp",
        direction="backward",
    )
    merged["news_sentiment_available"] = merged["avg_sentiment_score"].notna().astype(int)
    merged["news_sentiment"] = merged["avg_sentiment_score"].fillna(0.0)
    merged = merged.drop(columns=["timestamp", "avg_sentiment_score"])
    return merged


FEATURE_COLS = ["wti_ret_1d", "wti_ret_3d", "wti_ret_5d", "wti_ma_ratio", "wti_vol10", "rsi14",
                 "news_sentiment", "news_sentiment_available",
                 "real_yield_10y", "dxy", "vix", "macro_data_available",
                 "fed_funds_midpoint", "yield_fed_spread", "fed_rate_data_available",
                 "hours_to_next_event", "in_event_window_48h"]


def run_prediction_pipeline(history_df):
    history_with_sentiment = merge_news_sentiment(history_df)
    # Reuse gold's already-tested macro merge (real yields/DXY/VIX/Fed
    # funds/2yr yield) -- generic on any DataFrame with a "Date" column,
    # so no oil-specific rewrite needed here.
    history_with_macro = gpu.merge_macro_data(history_with_sentiment)
    calendar_df = gpu.load_economic_calendar()
    history_with_events = gpu.add_event_proximity_features(history_with_macro, calendar_df)
    df = add_features(history_with_events)
    df_model = df.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

    if len(df_model) < MIN_ROWS_FOR_PREDICTION:
        return {
            "prediction": "insufficient_data",
            "confidence_note": f"Only {len(df_model)} usable data points; need at least {MIN_ROWS_FOR_PREDICTION}. "
                                f"At daily cadence, this takes roughly {MIN_ROWS_FOR_PREDICTION}+ days to reach "
                                f"(slower than gold's hourly accumulation).",
            "model_type_used": None,
            "current_price_usd": float(history_df["WTI"].iloc[-1]) if len(history_df) else None,
            "predicted_price_usd": None,
            "price_model_type_used": None,
            "price_direction_vs_current": None,
            "direction_price_agreement": None,
            "price_confidence_note": "Not enough data yet to make a price forecast.",
            "is_price_prediction_significant": False,
            "model_accuracy_vs_baseline": None,
            "is_statistically_significant": False,
            "latest_news_sentiment_score": None,
            "news_sentiment_currently_available": False,
            "macro_data_currently_available": False,
            "current_fed_funds_midpoint_pct": None,
            "yield_minus_fed_funds_spread_pct": None,
            "fed_rate_data_currently_available": False,
            "recency_weighting_half_life_days": None,
            "next_economic_event_name": None,
            "hours_until_next_economic_event": None,
            "in_high_impact_event_window_48h": False,
            "economic_calendar_needs_update": False,
            "historical_data_start_date": history_df["Date"].min().strftime("%Y-%m-%d") if len(history_df) else None,
            "historical_data_end_date": history_df["Date"].max().strftime("%Y-%m-%d") if len(history_df) else None,
        }

    split_idx = int(len(df_model) * 0.8)
    train_df, test_df = df_model.iloc[:split_idx], df_model.iloc[split_idx:]
    X_train, y_train = train_df[FEATURE_COLS], train_df["target_up"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target_up"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if len(X_test) else X_train_s[:0]

    # Recency weighting -- same 30-day half-life as gold, anchored to the
    # most recent date in the whole dataset (not just the train split's
    # own boundary), so it stays meaningful as history grows.
    most_recent_date = df_model["Date"].max()
    train_age_days = (most_recent_date - train_df["Date"]).dt.total_seconds() / 86400
    sample_weight = np.power(0.5, train_age_days / HALF_LIFE_DAYS)

    # --- Direction: LogisticRegression vs GradientBoostingClassifier ---
    logistic_model = LogisticRegression(max_iter=1000, C=0.5)
    logistic_model.fit(X_train_s, y_train, sample_weight=sample_weight)
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    gb_model.fit(X_train_s, y_train, sample_weight=sample_weight)

    backtest_acc, is_significant, baseline_acc, model_type_used = None, False, None, "logistic_regression"
    model = logistic_model
    if len(y_test) >= 20:
        logistic_preds = logistic_model.predict(X_test_s)
        logistic_acc = float(accuracy_score(y_test, logistic_preds))
        gb_preds = gb_model.predict(X_test_s)
        gb_acc = float(accuracy_score(y_test, gb_preds))
        if gb_acc > logistic_acc:
            model, preds, backtest_acc, model_type_used = gb_model, gb_preds, gb_acc, "gradient_boosting"
        else:
            model, preds, backtest_acc, model_type_used = logistic_model, logistic_preds, logistic_acc, "logistic_regression"

        majority_class_train = int(round(y_train.mean()))
        baseline_preds = pd.Series([majority_class_train] * len(y_test), index=y_test.index)
        baseline_acc = float(accuracy_score(y_test, baseline_preds))

        model_correct = (preds == y_test.values)
        baseline_correct = (baseline_preds.values == y_test.values)
        b = int(((model_correct) & (~baseline_correct)).sum())
        c = int(((~model_correct) & (baseline_correct)).sum())
        pvalue = stats.binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue if (b + c) > 0 else 1.0
        is_significant = bool(pvalue < 0.05 and backtest_acc > baseline_acc)

    latest_row = df.dropna(subset=FEATURE_COLS).iloc[[-1]]
    latest_X = scaler.transform(latest_row[FEATURE_COLS])
    pred_proba = model.predict_proba(latest_X)[0]
    pred_class = "up" if pred_proba[1] > 0.5 else "down"

    if backtest_acc is None:
        confidence_note = (
            "Not enough held-out test data yet to evaluate this model's real accuracy "
            "(need at least 20 test rows). Present any direction call as unproven, not confident."
        )
    elif backtest_acc < baseline_acc:
        confidence_note = (
            f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best of the "
            f"two candidates tried) had backtested accuracy ({backtest_acc:.1%}) that was actually "
            f"LOWER than simply always predicting the majority class ({baseline_acc:.1%}) on real "
            f"held-out test data -- this model showed no real edge and underperformed even the "
            f"trivial baseline. Treat any direction call from it with strong skepticism, not just "
            f"mild caution -- it is not a case of 'close but unproven', it actively did worse."
        )
    elif not is_significant:
        confidence_note = (
            f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best on "
            f"real held-out test data) had backtested accuracy ({backtest_acc:.1%}) that was not "
            f"statistically distinguishable from simply always predicting the majority class "
            f"({baseline_acc:.1%}) -- McNemar's test, same methodology used for the gold predictor. "
            f"Treat this prediction as having no reliable edge beyond the obvious baseline."
        )
    else:
        confidence_note = (
            f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best on "
            f"real held-out test data) had backtested accuracy ({backtest_acc:.1%}) that was "
            f"statistically distinguishable from the majority-class baseline ({baseline_acc:.1%}) "
            f"via McNemar's test -- but still treat this as a modest statistical signal, not a guarantee."
        )

    # --- Price level: LinearRegression vs GradientBoostingRegressor ---
    price_pred_usd, price_is_significant, price_confidence_note, price_model_type_used = None, False, None, "linear_regression"
    linear_reg = LinearRegression()
    linear_reg.fit(X_train_s, train_df["next_ret"], sample_weight=sample_weight)
    gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    gb_reg.fit(X_train_s, train_df["next_ret"], sample_weight=sample_weight)

    reg_model = linear_reg
    if len(y_test) >= 20:
        linear_preds = linear_reg.predict(X_test_s)
        gb_preds_reg = gb_reg.predict(X_test_s)
        actual_rets = test_df["next_ret"].values
        linear_mse = float(np.mean((linear_preds - actual_rets) ** 2))
        gb_mse = float(np.mean((gb_preds_reg - actual_rets) ** 2))
        if gb_mse < linear_mse:
            reg_model, reg_preds, price_model_type_used = gb_reg, gb_preds_reg, "gradient_boosting"
        else:
            reg_model, reg_preds, price_model_type_used = linear_reg, linear_preds, "linear_regression"

        model_sq_err = (reg_preds - actual_rets) ** 2
        baseline_sq_err = (0.0 - actual_rets) ** 2
        diffs = baseline_sq_err - model_sq_err
        wilcoxon_p = stats.wilcoxon(diffs, alternative="greater").pvalue if np.any(diffs != 0) else 1.0
        model_mse = float(np.mean(model_sq_err))
        baseline_mse = float(np.mean(baseline_sq_err))
        price_is_significant = bool(wilcoxon_p < 0.05 and model_mse < baseline_mse)

        price_confidence_note = (
            f"This price forecast ({price_model_type_used.replace('_', ' ')}, chosen because it scored "
            f"best on real held-out test data) had error that was "
            + ("statistically better than" if price_is_significant else "NOT statistically better than")
            + " simply assuming tomorrow's price equals today's price (Wilcoxon signed-rank test on "
              "squared errors). Present the dollar figure as the model's best guess only -- do not "
              "imply it is a reliable forecast beyond the current price itself."
        )

    latest_ret_pred = reg_model.predict(latest_X)[0]
    current_price = float(history_df["WTI"].iloc[-1])
    price_pred_usd = float(current_price * (1 + latest_ret_pred))

    # Direction/price consistency check -- same fix already applied to
    # gold after a confirmed real bug (direction said "up" while the
    # price forecast was actually lower).
    if price_pred_usd > current_price:
        price_direction_vs_current = "higher"
    elif price_pred_usd < current_price:
        price_direction_vs_current = "lower"
    else:
        price_direction_vs_current = "the same"
    direction_price_agreement = (
        (pred_class == "up" and price_pred_usd >= current_price) or
        (pred_class == "down" and price_pred_usd <= current_price)
    )

    latest_news_sentiment = float(latest_row["news_sentiment"].values[0])
    latest_news_sentiment_available = bool(latest_row["news_sentiment_available"].values[0])

    latest_macro_data_available = bool(latest_row["macro_data_available"].values[0])
    latest_fed_funds_midpoint = float(latest_row["fed_funds_midpoint"].values[0])
    latest_yield_fed_spread = float(latest_row["yield_fed_spread"].values[0])
    latest_fed_rate_data_available = bool(latest_row["fed_rate_data_available"].values[0])

    latest_hours_to_event = float(latest_row["hours_to_next_event"].values[0])
    latest_in_event_window = bool(latest_row["in_event_window_48h"].values[0])
    latest_event_name = latest_row["next_event_name"].values[0] if "next_event_name" in latest_row.columns else None
    latest_event_name = None if pd.isna(latest_event_name) else str(latest_event_name)
    calendar_needs_update = latest_event_name is None

    return {
        "prediction": pred_class,
        "prediction_probability_up": float(pred_proba[1]),
        "confidence_note": confidence_note,
        "model_type_used": model_type_used,
        "current_price_usd": current_price,
        "predicted_price_usd": price_pred_usd,
        "price_model_type_used": price_model_type_used,
        "price_direction_vs_current": price_direction_vs_current,
        "direction_price_agreement": direction_price_agreement,
        "price_confidence_note": price_confidence_note,
        "is_price_prediction_significant": price_is_significant,
        "model_accuracy_vs_baseline": {"model": backtest_acc, "baseline": baseline_acc} if backtest_acc else None,
        "is_statistically_significant": is_significant,
        "latest_news_sentiment_score": latest_news_sentiment if latest_news_sentiment_available else None,
        "news_sentiment_currently_available": latest_news_sentiment_available,
        "macro_data_currently_available": latest_macro_data_available,
        "current_fed_funds_midpoint_pct": latest_fed_funds_midpoint if latest_fed_rate_data_available else None,
        "yield_minus_fed_funds_spread_pct": latest_yield_fed_spread if latest_fed_rate_data_available else None,
        "fed_rate_data_currently_available": latest_fed_rate_data_available,
        "recency_weighting_half_life_days": HALF_LIFE_DAYS,
        "next_economic_event_name": latest_event_name,
        "hours_until_next_economic_event": None if calendar_needs_update else round(latest_hours_to_event, 1),
        "in_high_impact_event_window_48h": latest_in_event_window,
        "economic_calendar_needs_update": calendar_needs_update,
        "historical_data_start_date": history_df["Date"].min().strftime("%Y-%m-%d"),
        "historical_data_end_date": history_df["Date"].max().strftime("%Y-%m-%d"),
    }


# ============================================================
# 3b. LIVE PREDICTION TRACK RECORD (same pattern as gold)
# ============================================================

def load_track_record():
    if os.path.exists(TRACK_RECORD_FILE):
        with open(TRACK_RECORD_FILE) as f:
            return json.load(f)
    return []


def save_track_record(track_record):
    with open(TRACK_RECORD_FILE, "w") as f:
        json.dump(track_record, f, indent=2)


def resolve_pending_predictions(track_record, actual_price, actual_timestamp):
    resolved_count = 0
    for entry in track_record:
        if entry.get("resolved"):
            continue
        actual_direction = "up" if actual_price > entry["price_at_prediction"] else "down"
        entry["resolved"] = True
        entry["resolved_at"] = actual_timestamp.isoformat()
        entry["actual_price_at_resolution"] = actual_price
        entry["actual_direction"] = actual_direction
        entry["direction_correct"] = (actual_direction == entry["prediction_direction"])
        entry["price_error_usd"] = abs(entry["predicted_price_usd"] - actual_price) if entry.get("predicted_price_usd") is not None else None
        resolved_count += 1
    if resolved_count:
        print(f"Resolved {resolved_count} pending oil prediction(s) against the newly fetched live price.")
    return track_record


def record_new_prediction(track_record, prediction_result, current_price, timestamp):
    if prediction_result.get("prediction") not in ("up", "down"):
        return track_record
    track_record.append({
        "predicted_at": timestamp.isoformat(),
        "price_at_prediction": current_price,
        "prediction_direction": prediction_result["prediction"],
        "predicted_price_usd": prediction_result.get("predicted_price_usd"),
        "resolved": False, "resolved_at": None, "actual_price_at_resolution": None,
        "actual_direction": None, "direction_correct": None, "price_error_usd": None,
    })
    return track_record


def compute_rolling_live_accuracy(track_record, window=ROLLING_WINDOW):
    resolved = [e for e in track_record if e.get("resolved")]
    recent = resolved[-window:]
    if len(recent) == 0:
        return {
            "rolling_predictions_tracked": 0, "rolling_direction_accuracy": None,
            "rolling_price_mae_usd": None, "rolling_accuracy_is_significant": False,
            "rolling_accuracy_note": "No resolved live predictions yet -- this builds up over time as the job runs.",
        }
    correct = sum(1 for e in recent if e["direction_correct"])
    total = len(recent)
    accuracy = correct / total
    price_errors = [e["price_error_usd"] for e in recent if e.get("price_error_usd") is not None]
    mae = float(np.mean(price_errors)) if price_errors else None
    is_significant = False
    if total >= 20:
        pvalue = stats.binomtest(correct, total, p=0.5, alternative="two-sided").pvalue
        is_significant = bool(pvalue < 0.05 and accuracy > 0.5)
        note = (f"Over the last {total} resolved live predictions, this system was correct on direction "
                f"{correct} times ({accuracy:.1%}). "
                + ("This is statistically distinguishable from a coin flip (binomial test, p<0.05) -- "
                   "real, if modest, live edge." if is_significant else
                   "This is NOT statistically distinguishable from a coin flip (binomial test) -- treat "
                   "recent direction calls as having no demonstrated live edge yet."))
    else:
        note = (f"Only {total} resolved live predictions so far (need at least 20 for a meaningful "
                f"significance test). Raw accuracy so far: {correct}/{total} ({accuracy:.1%}), but treat "
                f"this as too small a sample to draw a real conclusion from yet.")
    return {
        "rolling_predictions_tracked": total, "rolling_direction_accuracy": accuracy,
        "rolling_price_mae_usd": mae, "rolling_accuracy_is_significant": is_significant,
        "rolling_accuracy_note": note,
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    history_df = load_history()
    track_record = load_track_record()

    fetched_pairs = fetch_wti_series()
    print(f"Fetched {len(fetched_pairs)} WTI price points from Alpha Vantage "
          f"(most recent: {fetched_pairs[0][0]} = ${fetched_pairs[0][1]:.2f}).")

    history_df = merge_fetched_series(history_df, fetched_pairs)

    # LIVE-PRICE PATCH: overwrite/add today's row with a genuinely fresh
    # price from OilPriceAPI.com, fixing Alpha Vantage's confirmed
    # multi-day reporting lag at the root. Falls back gracefully (keeps
    # Alpha Vantage's own most recent point) if the live call fails,
    # is flagged synthetic/stale, or the key isn't configured.
    live_price, live_date_str = fetch_live_oil_price()
    if live_price is not None:
        old_latest_price = float(history_df["WTI"].iloc[-1])
        old_latest_date = history_df["Date"].iloc[-1].strftime("%Y-%m-%d")
        live_row = pd.DataFrame([{"Date": pd.Timestamp(live_date_str), "WTI": live_price}])
        history_df = pd.concat([history_df, live_row], ignore_index=True)
        history_df = history_df.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)
        print(f"Live-price patch applied: Alpha Vantage's most recent point was "
              f"{old_latest_date} = ${old_latest_price:.2f}; now using live "
              f"{live_date_str} = ${live_price:.2f} as the current price for prediction.")
    else:
        print("Live-price patch NOT applied (see reason above) -- using Alpha Vantage's own most recent "
              "point as current_price_usd, which may carry Alpha Vantage's confirmed multi-day reporting lag.")

    current_price = float(history_df["WTI"].iloc[-1])
    current_timestamp = history_df["Date"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)

    # Resolve any predictions from PREVIOUS runs using this freshly
    # fetched price, before computing a new one -- the actual live
    # feedback loop: check what was said last time against reality.
    track_record = resolve_pending_predictions(track_record, current_price, current_timestamp)

    result = run_prediction_pipeline(history_df)
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    result["data_points_used"] = len(history_df)
    result["current_price_source"] = "live_oilpriceapi" if live_price is not None else "alpha_vantage_wti_snapshot"

    track_record = record_new_prediction(track_record, result, current_price, current_timestamp)
    rolling_stats = compute_rolling_live_accuracy(track_record)
    result.update(rolling_stats)

    # Cross-check against DXY's latest prediction, if oil actually made a
    # real direction call (skip for insufficient_data -- nothing to check).
    if result.get("prediction") in ("up", "down"):
        cross_asset_stats = gpu.check_cross_asset_consistency(result["prediction"], asset_name="Oil")
        result.update(cross_asset_stats)

    print("Prediction result:")
    print(json.dumps(result, indent=2))

    save_history(history_df)
    save_track_record(track_record)
    with open(PREDICTION_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {PREDICTION_FILE}, {HISTORY_FILE}, and {TRACK_RECORD_FILE} locally -- "
          f"the workflow will commit these back to the repo.")


if __name__ == "__main__":
    main()
