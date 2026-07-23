"""
gold_predictor_updater.py
===========================

Run this on a schedule (GitHub Actions, every 8 hours) to keep the gold
price prediction fresh, using the free GoldAPI.io tier (100 requests/month).

ARCHITECTURE (corrected a second time, after a real production issue)
-------------------------------------------------------------------------
The previous version POSTed results to the PHP/Node backend running on
Render, which stored them as a local file. That broke in practice:
Render's FREE web services don't just lose local files on redeploy --
they lose them every time the service goes to sleep from inactivity
(15 minutes) and wakes back up again, which happens routinely for a
low-traffic site. The prediction data would vanish until the next cron
run, up to 8 hours later.

The fix: this script now reads and writes the prediction/history data as
plain files IN THIS SAME REPOSITORY (gold-predictor). The GitHub Actions
workflow that runs this script also commits the updated files back to the
repo afterward (see update_gold_prediction.yml) -- GitHub's own storage
is genuinely persistent, unlike Render's free-tier ephemeral disk. The
chatbot backend then reads the latest prediction directly from this
repo's raw GitHub URL on every request, instead of relying on its own
fragile local copy. This also removes the need for the two custom
backend endpoints, the shared-secret auth, and simplifies the whole
system.

SETUP REQUIRED
-----------------
Set this environment variable on the GitHub Actions workflow (already
done as a repository secret if you followed the setup steps):
    (No API key needed anymore -- Binance's public endpoint requires none.)

No backend URLs or shared secret are needed anymore.
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

API_KEY = os.environ.get("GOLDPRICEZ_API_KEY", "")   # your GoldPriceZ.com key
SEED_CSV = "gld_price_data.csv"                    # bundled with this script; used only on first-ever run
HISTORY_FILE = "gold_price_history.json"            # lives in this repo, committed back after each run
PREDICTION_FILE = "gold_prediction_latest.json"     # lives in this repo, committed back after each run
MIN_ROWS_FOR_PREDICTION = 60


# ============================================================
# 1. LIVE PRICE (confirmed format against GoldPriceZ.com's real, full
#    API documentation page -- see the project's build notes)
# ============================================================

def fetch_live_price():
    """Fetch the current gold spot price from GoldPriceZ.com.

    HISTORY OF THIS FUNCTION, briefly: started on GoldAPI.io (100
    requests/month free tier, reliable from GitHub Actions); tried
    switching to Binance's public API for higher frequency, which failed
    -- Binance blocks automated/datacenter traffic, confirmed via a real
    HTTP 451 error from GitHub Actions' servers specifically. Now on
    GoldPriceZ.com instead: 60 requests/hour confirmed directly by their
    support team, reliable from GitHub Actions, no geo/datacenter
    blocking encountered. Requires a visible attribution link on the
    site's homepage per their terms (added, and confirmed with their
    support team before they activated the key)."""
    if not API_KEY:
        raise RuntimeError("GOLDPRICEZ_API_KEY is not set.")
    url = "https://goldpricez.com/api/rates/currency/usd/measure/ounce"
    headers = {"X-API-KEY": API_KEY}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    # GoldPriceZ's response is double-encoded JSON: the raw body is a JSON
    # STRING containing another JSON object as escaped text, not a plain
    # JSON object directly. Confirmed directly against a real captured
    # response (raw text looked like "{\"ounce_price_usd\":\"...\",...}"
    # -- note the outer quotes). A single resp.json() call correctly
    # un-escapes the outer string layer but returns a Python str, not a
    # dict; parsing that string again with json.loads() gets the real
    # dict. This is unusual but confirmed real, not a guess.
    outer = resp.json()
    data = json.loads(outer) if isinstance(outer, str) else outer
    price = float(data["ounce_price_usd"])
    return price, datetime.now(timezone.utc)


# ============================================================
# 2. LOCAL HISTORY FILE (lives in the repo; committed back by the
#    workflow after each run, so it persists across runs for real)
# ============================================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            records = json.load(f)
        if records:
            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            return df[["Date", "GLD"]].sort_values("Date").reset_index(drop=True)
    # First-ever run, or empty file: start with NO seed data.
    #
    # IMPORTANT: an earlier version of this script seeded from a bundled
    # CSV of 2008-2018 GLD ETF *share prices* (~$85-130). That is a
    # different unit/instrument than GoldAPI's live feed, which returns
    # *spot price per troy ounce* (~$4000+). Mixing the two produced a
    # nonsensical prediction (probability ~1e-153) that was technically
    # still caught by the significance test but was a meaningless
    # computation, not just an insignificant one. Rather than patch that
    # over with a rough conversion factor (gold's ETF-share-to-spot ratio
    # drifts over time and isn't reliable to assume), we start with an
    # honestly empty history and let it accumulate from real, consistent
    # live GoldAPI prices only. At 3 points/day (every 8 hours), this
    # reaches MIN_ROWS_FOR_PREDICTION (60) in about 20 days. Until then,
    # the pipeline correctly and honestly returns "insufficient_data"
    # rather than a computed-but-meaningless number.
    print("No history file found or it was empty -- starting fresh with no seed data (see comment above for why).")
    return pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "GLD": pd.Series(dtype="float64")})


def save_history(history_df):
    records = history_df.copy()
    records["Date"] = pd.to_datetime(records["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, "w") as f:
        json.dump(records.to_dict(orient="records"), f)


# ============================================================
# 3. PREDICTION PIPELINE (unchanged, already validated)
# ============================================================

def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["gld_ret_1d"] = df["GLD"].pct_change(1)
    df["gld_ret_3d"] = df["GLD"].pct_change(3)
    df["gld_ret_5d"] = df["GLD"].pct_change(5)
    df["gld_ma5"] = df["GLD"].rolling(5).mean()
    df["gld_ma20"] = df["GLD"].rolling(20).mean()
    df["gld_ma_ratio"] = df["gld_ma5"] / df["gld_ma20"]
    df["gld_vol10"] = df["gld_ret_1d"].rolling(10).std()
    delta = df["GLD"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["next_ret"] = df["GLD"].shift(-1) / df["GLD"] - 1
    df["target_up"] = (df["next_ret"] > 0).astype(int)
    return df


def merge_news_sentiment(price_df, sentiment_file="news_sentiment_history.json"):
    """Attach a news-sentiment feature to each price row, using only
    sentiment readings that existed AT OR BEFORE that price row's
    timestamp -- critical to avoid lookahead bias (a later news reading
    leaking into an earlier prediction, which would silently invalidate
    every backtest and significance test in this pipeline). Implemented
    via pandas merge_asof with direction='backward', which is exactly
    designed for this "most recent known value as of time T" alignment.

    Missing/no-relevant-articles periods and the period before the first
    real sentiment reading are filled with neutral (0.0), with a separate
    flag column so the model can distinguish "genuinely neutral news" from
    "no sentiment data was available yet" if that distinction matters."""
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
        direction="backward",  # only ever look at PAST sentiment readings, never future ones
    )
    merged["news_sentiment_available"] = merged["avg_sentiment_score"].notna().astype(int)
    merged["news_sentiment"] = merged["avg_sentiment_score"].fillna(0.0)
    merged = merged.drop(columns=["timestamp", "avg_sentiment_score"])
    return merged


FEATURE_COLS = ["gld_ret_1d", "gld_ret_3d", "gld_ret_5d", "gld_ma_ratio", "gld_vol10", "rsi14",
                 "news_sentiment", "news_sentiment_available"]


def run_prediction_pipeline(history_df):
    history_with_sentiment = merge_news_sentiment(history_df)
    df = add_features(history_with_sentiment)
    df_model = df.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

    if len(df_model) < MIN_ROWS_FOR_PREDICTION:
        return {
            "prediction": "insufficient_data",
            "confidence_note": f"Only {len(df_model)} usable data points; need at least {MIN_ROWS_FOR_PREDICTION}.",
            "current_price_usd": float(history_df["GLD"].iloc[-1]) if len(history_df) else None,
            "predicted_price_usd": None,
            "price_confidence_note": "Not enough data yet to make a price forecast.",
            "is_price_prediction_significant": False,
            "model_accuracy_vs_baseline": None,
            "is_statistically_significant": False,
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

        # Baseline: always predict the majority class observed in TRAINING
        # data (matches what a naive practitioner would actually do; using
        # the test set's own majority fraction here would leak information).
        majority_class_train = int(round(y_train.mean()))
        baseline_preds = pd.Series([majority_class_train] * len(y_test), index=y_test.index)
        baseline_acc = float(accuracy_score(y_test, baseline_preds))

        # McNemar's test: compares the MODEL against the BASELINE directly
        # on the same test points, not against blind 50/50 chance. This
        # matters a lot when the outcome is imbalanced (e.g. gold went up
        # in ~98% of periods during a strong bull run in our own backfilled
        # data) -- in that situation, a trivial "always predict up" baseline
        # already scores ~98%, so testing the model against a coin flip
        # would call that "significant" even though the model added nothing
        # over the trivial baseline. An earlier version of this script did
        # exactly that and was caught and corrected here.
        model_correct = (preds == y_test.values)
        baseline_correct = (baseline_preds.values == y_test.values)
        # b = model right, baseline wrong; c = model wrong, baseline right
        b = int(((model_correct) & (~baseline_correct)).sum())
        c = int(((~model_correct) & (baseline_correct)).sum())
        if b + c > 0:
            pvalue = stats.binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue
        else:
            pvalue = 1.0  # model and baseline never disagreed -- definitely not distinguishable
        is_significant = bool(pvalue < 0.05 and backtest_acc > baseline_acc)

    latest_row = df.dropna(subset=FEATURE_COLS).iloc[[-1]]
    latest_X = scaler.transform(latest_row[FEATURE_COLS])
    pred_proba = model.predict_proba(latest_X)[0]
    pred_class = "up" if pred_proba[1] > 0.5 else "down"

    if not is_significant:
        confidence_note = (
            f"This model's backtested accuracy ({backtest_acc:.1%}) was not statistically "
            f"distinguishable from simply always predicting the majority class "
            f"({baseline_acc:.1%}) -- McNemar's test, not a comparison to blind chance, since "
            f"the outcome can be imbalanced (e.g. during a strong sustained trend). Treat this "
            f"prediction as having no reliable edge beyond the obvious baseline -- state that "
            f"plainly rather than presenting the direction confidently."
        )
    else:
        confidence_note = (
            f"Backtested accuracy ({backtest_acc:.1%}) was statistically distinguishable "
            f"from the majority-class baseline ({baseline_acc:.1%}) via McNemar's test, "
            f"meaning the model adds real signal beyond the trivial baseline -- but this is "
            f"still a modest statistical signal, not a guarantee -- present it with "
            f"appropriate caveats."
        )

    # --------------------------------------------------------------
    # PRICE-LEVEL PREDICTION (a real USD number, not just direction)
    # --------------------------------------------------------------
    # IMPORTANT: naively predicting tomorrow's PRICE LEVEL is a well-known
    # trap -- "tomorrow's price ~= today's price" already looks highly
    # accurate (high R^2) purely because prices are autocorrelated, without
    # reflecting any real skill. This is the same issue flagged with the
    # original dataset's misleading "R^2=0.98" claim at the very start of
    # this project. To give an honest dollar figure instead of a
    # misleadingly precise-looking one, we: (1) predict the RETURN
    # (percentage change), not the raw price level, using a real regression
    # model; (2) apply that predicted return to the current price to get a
    # dollar figure; (3) test the regression's prediction error against the
    # trivial "no change" baseline (predicting 0% return, i.e. tomorrow's
    # price = today's price) using a paired Wilcoxon signed-rank test on
    # squared errors -- the regression analogue of the McNemar's test used
    # above for direction, so the same honesty standard applies to both.
    price_pred_usd, price_is_significant, price_confidence_note = None, False, None
    reg_model = LinearRegression()
    reg_model.fit(X_train_s, train_df["next_ret"])

    if len(y_test) >= 20:
        reg_preds = reg_model.predict(X_test_s)
        actual_rets = test_df["next_ret"].values
        model_sq_err = (reg_preds - actual_rets) ** 2
        baseline_sq_err = (0.0 - actual_rets) ** 2  # naive "no change" baseline
        diffs = baseline_sq_err - model_sq_err  # positive = model beat baseline on that point
        if np.any(diffs != 0):
            wilcoxon_p = stats.wilcoxon(diffs, alternative="greater").pvalue
        else:
            wilcoxon_p = 1.0
        model_mse = float(np.mean(model_sq_err))
        baseline_mse = float(np.mean(baseline_sq_err))
        price_is_significant = bool(wilcoxon_p < 0.05 and model_mse < baseline_mse)

        if not price_is_significant:
            price_confidence_note = (
                f"This price forecast's error was not statistically better than simply "
                f"assuming tomorrow's price equals today's price (Wilcoxon signed-rank test "
                f"on squared errors). Present the dollar figure as the model's best guess only "
                f"-- do not imply it is a reliable forecast beyond the current price itself."
            )
        else:
            price_confidence_note = (
                f"This price forecast's error was statistically better than the naive "
                f"'no change' baseline (Wilcoxon signed-rank test), meaning the regression "
                f"adds real signal -- but treat the exact dollar figure as an estimate with "
                f"real uncertainty, not a precise forecast."
            )

    latest_ret_pred = reg_model.predict(latest_X)[0]
    current_price = float(history_df["GLD"].iloc[-1])
    price_pred_usd = float(current_price * (1 + latest_ret_pred))

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
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    history_df = load_history()

    price, timestamp = fetch_live_price()
    print(f"Fetched live price: ${price:.2f} at {timestamp.isoformat()}")

    live_row = pd.DataFrame([{"Date": pd.Timestamp(timestamp).tz_localize(None), "GLD": price}])
    history_df = pd.concat([history_df, live_row], ignore_index=True)
    history_df = history_df.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)

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
