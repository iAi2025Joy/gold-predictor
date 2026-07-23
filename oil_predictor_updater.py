"""
oil_predictor_updater.py
============================

Oil-market counterpart to gold_predictor_updater.py, same architecture and
same honesty standards. Run on a schedule (GitHub Actions, DAILY -- not
hourly, see cadence note below) to keep an oil price prediction fresh.

*** VERIFICATION STATUS: PARSING LOGIC UNVERIFIED AGAINST A REAL KEY ***
This was written against Alpha Vantage's PUBLICLY DOCUMENTED response shape
for their "economic indicator"-style endpoints (the same {"data": [{"date":
..., "value": ...}]} shape used by their REAL_GDP, CPI, and Treasury Yield
endpoints, which the WTI/BRENT commodity endpoints are documented under the
same family as). The demo API key does NOT support the WTI function, so
this could not be tested against a real live response the way GoldPriceZ's
double-encoded-JSON quirk was caught and confirmed before writing code
against it. TREAT THIS AS UNVERIFIED until run once for real with a real
ALPHAVANTAGE_API_KEY -- check the printed raw response on first run, and
fix parse_wti_response() below if the real shape differs. Do not trust
silently-succeeding output from this script until that first real check
has happened.

IMPORTANT DESIGN NOTE -- FULL-SERIES MERGE, NOT SINGLE-POINT APPEND:
Endpoints in this same Alpha Vantage family (REAL_GDP, CPI, Treasury Yield)
are documented to return their ENTIRE historical "data" array in every
response, not just the latest value -- this is a real, meaningful
difference from GoldPriceZ (which only ever returns a single current spot
price, requiring gold's history to accumulate one point per call). If WTI
behaves the same way (to be confirmed on first real run), that means a
SINGLE call can backfill a full year or more of history immediately,
instead of waiting 60+ days for daily accumulation to clear the prediction
threshold. So this script does NOT just fetch-and-append-latest like the
gold predictor does -- it fetches the full returned series EVERY run and
MERGES it into local history by date (freshly-fetched values win on any
date collision, since they're the canonical source). This also means a
missed day (e.g. a failed Actions run) self-heals automatically next run,
which gold's single-point-append approach cannot do.

CADENCE: still runs DAILY (matches the WTI endpoint's daily granularity) --
merging the full series more than once a day would just re-fetch the same
data repeatedly for no benefit.

BUDGET: reuses the SAME ALPHAVANTAGE_API_KEY already used by
news_sentiment_fetcher.py, sharing the same 25 requests/day total budget.
Still exactly 1 request per run.
"""

import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")  # SAME key as news_sentiment_fetcher.py
HISTORY_FILE = "oil_price_history.json"           # lives in this repo, committed back after each run
PREDICTION_FILE = "oil_prediction_latest.json"     # lives in this repo, committed back after each run
MIN_ROWS_FOR_PREDICTION = 60


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
                 "news_sentiment", "news_sentiment_available"]


def run_prediction_pipeline(history_df):
    history_with_sentiment = merge_news_sentiment(history_df)
    df = add_features(history_with_sentiment)
    df_model = df.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

    if len(df_model) < MIN_ROWS_FOR_PREDICTION:
        return {
            "prediction": "insufficient_data",
            "confidence_note": f"Only {len(df_model)} usable data points; need at least {MIN_ROWS_FOR_PREDICTION}. "
                                f"At daily cadence, this takes roughly {MIN_ROWS_FOR_PREDICTION}+ days to reach "
                                f"(slower than gold's hourly accumulation).",
            "current_price_usd": float(history_df["WTI"].iloc[-1]) if len(history_df) else None,
            "predicted_price_usd": None,
            "price_confidence_note": "Not enough data yet to make a price forecast.",
            "is_price_prediction_significant": False,
            "model_accuracy_vs_baseline": None,
            "is_statistically_significant": False,
            "latest_news_sentiment_score": None,
            "news_sentiment_currently_available": False,
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

    model = LogisticRegression(max_iter=1000, C=0.5)
    model.fit(X_train_s, y_train)

    backtest_acc, is_significant, baseline_acc = None, False, None
    if len(y_test) >= 20:
        preds = model.predict(X_test_s)
        backtest_acc = float(accuracy_score(y_test, preds))
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
            "Not enough test-set data points yet to run a reliable significance test "
            "(need at least 20 held-out points) -- direction is reported below, but with "
            "no accuracy/significance backing it yet. Treat as very low confidence."
        )
    elif not is_significant:
        confidence_note = (
            f"This model's backtested accuracy ({backtest_acc:.1%}) was not statistically "
            f"distinguishable from simply always predicting the majority class ({baseline_acc:.1%}) -- "
            f"McNemar's test, same methodology used for the gold predictor. Treat this prediction as "
            f"having no reliable edge beyond the obvious baseline."
        )
    else:
        confidence_note = (
            f"Backtested accuracy ({backtest_acc:.1%}) was statistically distinguishable from the "
            f"majority-class baseline ({baseline_acc:.1%}) via McNemar's test -- but still treat this "
            f"as a modest statistical signal, not a guarantee."
        )

    price_pred_usd, price_is_significant, price_confidence_note = None, False, None
    reg_model = LinearRegression()
    reg_model.fit(X_train_s, train_df["next_ret"])

    if len(y_test) >= 20:
        reg_preds = reg_model.predict(X_test_s)
        actual_rets = test_df["next_ret"].values
        model_sq_err = (reg_preds - actual_rets) ** 2
        baseline_sq_err = (0.0 - actual_rets) ** 2
        diffs = baseline_sq_err - model_sq_err
        wilcoxon_p = stats.wilcoxon(diffs, alternative="greater").pvalue if np.any(diffs != 0) else 1.0
        model_mse = float(np.mean(model_sq_err))
        baseline_mse = float(np.mean(baseline_sq_err))
        price_is_significant = bool(wilcoxon_p < 0.05 and model_mse < baseline_mse)

        price_confidence_note = (
            "This price forecast's error was not statistically better than assuming tomorrow's price "
            "equals today's price (Wilcoxon signed-rank test). Present the dollar figure as the model's "
            "best guess only." if not price_is_significant else
            "This price forecast's error was statistically better than the naive 'no change' baseline -- "
            "treat the exact figure as an estimate with real uncertainty, not a precise forecast."
        )

    latest_ret_pred = reg_model.predict(latest_X)[0]
    current_price = float(history_df["WTI"].iloc[-1])
    price_pred_usd = float(current_price * (1 + latest_ret_pred))

    latest_news_sentiment = float(latest_row["news_sentiment"].values[0])
    latest_news_sentiment_available = bool(latest_row["news_sentiment_available"].values[0])

    return {
        "prediction": pred_class,
        "prediction_probability_up": float(pred_proba[1]),
        "confidence_note": confidence_note,
        "current_price_usd": current_price,
        "predicted_price_usd": price_pred_usd,
        "price_confidence_note": price_confidence_note,
        "is_price_prediction_significant": price_is_significant,
        "model_accuracy_vs_baseline": {"model": backtest_acc, "baseline": baseline_acc} if backtest_acc else None,
        "is_statistically_significant": is_significant,
        "latest_news_sentiment_score": latest_news_sentiment if latest_news_sentiment_available else None,
        "news_sentiment_currently_available": latest_news_sentiment_available,
        "historical_data_start_date": history_df["Date"].min().strftime("%Y-%m-%d"),
        "historical_data_end_date": history_df["Date"].max().strftime("%Y-%m-%d"),
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    history_df = load_history()

    fetched_pairs = fetch_wti_series()
    print(f"Fetched {len(fetched_pairs)} WTI price points from Alpha Vantage "
          f"(most recent: {fetched_pairs[0][0]} = ${fetched_pairs[0][1]:.2f}).")
    print("*** If this is the first real run, VERIFY this looks like a real, plausible "
          "WTI series (dates in expected range, prices roughly $60-100/barrel for recent "
          "years) before trusting the pipeline -- see module docstring on unverified parsing. ***")

    history_df = merge_fetched_series(history_df, fetched_pairs)

    result = run_prediction_pipeline(history_df)
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    result["data_points_used"] = len(history_df)

    print("Prediction result:")
    print(json.dumps(result, indent=2))

    save_history(history_df)
    with open(PREDICTION_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {PREDICTION_FILE} and {HISTORY_FILE} locally -- the workflow will commit these back to the repo.")


if __name__ == "__main__":
    main()
