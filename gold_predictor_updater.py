"""
gold_predictor_updater.py
===========================

Run this on a schedule (Render Cron Job, every 8 hours) to keep the gold
price prediction fresh, using the free GoldAPI.io tier (100 requests/month).

ARCHITECTURE (corrected after seeing your real GoldAPI.io dashboard)
-------------------------------------------------------------------------
GoldAPI.io's free tier gives ONE price per request -- there is no bulk
"give me 5 years of history" endpoint like some paid services have.
Fetching years of daily history one request at a time would burn through
the 100/month budget instantly. So instead:

  1. Your local 2008-2018 dataset (gld_price_data.csv, bundled with this
     script) is the permanent SEED history -- fetched via the API zero
     times, it's just local data you already have.
  2. That seed lives on your PHP backend's own disk (the one piece of
     this system with real persistent storage -- see the architecture
     note in gold_prediction_function.php for why).
  3. Each cron run: fetch the CURRENT accumulated history from PHP, fetch
     ONE new live price from GoldAPI (1 of your 100 monthly requests),
     append it, run the prediction, POST the updated history + prediction
     back to PHP in a single call.
  4. On the very first run ever (PHP has no history yet), this script
     seeds PHP with the full local CSV instead of just one point.

At every-8-hours (3x/day), this uses about 90 of your 100 monthly
requests, leaving a small buffer for manual testing.

SETUP REQUIRED BEFORE THIS WILL WORK
--------------------------------------
1. Your free GoldAPI.io key (already obtained).
2. Set these environment variables on the Render Cron Job service:
       GOLD_API_KEY            -- your GoldAPI.io key (the x-access-token value)
       BACKEND_HISTORY_URL      -- e.g. https://ai-chat-backend-3-g573.onrender.com/gold-history
       BACKEND_UPDATE_URL       -- e.g. https://ai-chat-backend-3-g573.onrender.com/update-gold-prediction
       BACKEND_SHARED_SECRET    -- same random string set on your PHP backend
3. Bundle gld_price_data.csv alongside this script when deploying (same
   repo/folder), since it's the seed data for the very first run.
4. Create the Render Cron Job, schedule "0 */8 * * *" (every 8 hours).

WHAT I COULD AND COULD NOT TEST
-----------------------------------
The prediction pipeline (feature engineering, model, backtest, output
schema) is validated end-to-end using real historical data. The GoldAPI.io
live-price call now matches THREE independent confirmed sources including
your own dashboard, so I'm confident in that part specifically. The calls
to YOUR PHP backend's new /gold-history and /update-gold-prediction
endpoints could not be tested -- those don't exist until you deploy the
updated PHP file. Test the full round-trip once both sides are live, and
send me the exact error if anything doesn't match.
"""

import os
import json
from datetime import datetime, timezone

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ.get("GOLD_API_KEY", "")
BACKEND_HISTORY_URL = os.environ.get("BACKEND_HISTORY_URL", "")
BACKEND_UPDATE_URL = os.environ.get("BACKEND_UPDATE_URL", "")
BACKEND_SHARED_SECRET = os.environ.get("BACKEND_SHARED_SECRET", "")
SEED_CSV = "gld_price_data.csv"   # bundled with this script; used only on first-ever run
MIN_ROWS_FOR_PREDICTION = 60


# ============================================================
# 1. LIVE PRICE (confirmed format: 3 independent sources agree,
#    including your own GoldAPI.io dashboard)
# ============================================================

def fetch_live_price():
    if not API_KEY:
        raise RuntimeError("GOLD_API_KEY is not set.")
    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {"x-access-token": API_KEY, "Content-Type": "application/json"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    price = data["price"]  # confirmed top-level field from your real dashboard's sample response
    return float(price), datetime.now(timezone.utc)


# ============================================================
# 2. BACKEND HISTORY GET/POST (untested -- these endpoints don't
#    exist until gold_prediction_function.php is deployed with them)
# ============================================================

def fetch_history_from_backend():
    if not BACKEND_HISTORY_URL:
        raise RuntimeError("BACKEND_HISTORY_URL is not set.")
    headers = {"X-Shared-Secret": BACKEND_SHARED_SECRET}
    resp = requests.get(BACKEND_HISTORY_URL, headers=headers, timeout=15)
    if resp.status_code == 404:
        return None  # no history yet -- first-ever run
    resp.raise_for_status()
    records = resp.json()
    if not records:
        return None
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "GLD"]].sort_values("Date").reset_index(drop=True)


def load_seed_history():
    df = pd.read_csv(SEED_CSV, parse_dates=["Date"])
    return df[["Date", "GLD"]]


def push_history_and_prediction(history_df, prediction_result):
    if not BACKEND_UPDATE_URL:
        raise RuntimeError("BACKEND_UPDATE_URL is not set.")
    records = history_df.copy()
    records["Date"] = records["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        **prediction_result,
        "history": records.to_dict(orient="records"),
    }
    headers = {"Content-Type": "application/json", "X-Shared-Secret": BACKEND_SHARED_SECRET}
    resp = requests.post(BACKEND_UPDATE_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    print(f"Pushed to backend: {resp.status_code} {resp.text[:200]}")


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


FEATURE_COLS = ["gld_ret_1d", "gld_ret_3d", "gld_ret_5d", "gld_ma_ratio", "gld_vol10", "rsi14"]


def run_prediction_pipeline(history_df):
    df = add_features(history_df)
    df_model = df.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

    if len(df_model) < MIN_ROWS_FOR_PREDICTION:
        return {
            "prediction": "insufficient_data",
            "confidence_note": f"Only {len(df_model)} usable data points; need at least {MIN_ROWS_FOR_PREDICTION}.",
            "current_price_usd": float(history_df["GLD"].iloc[-1]) if len(history_df) else None,
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
        baseline_acc = float(max(y_test.mean(), 1 - y_test.mean()))
        n_correct = int((preds == y_test.values).sum())
        pvalue = stats.binomtest(n_correct, len(y_test), p=0.5, alternative="two-sided").pvalue
        is_significant = bool(pvalue < 0.05)

    latest_row = df.dropna(subset=FEATURE_COLS).iloc[[-1]]
    latest_X = scaler.transform(latest_row[FEATURE_COLS])
    pred_proba = model.predict_proba(latest_X)[0]
    pred_class = "up" if pred_proba[1] > 0.5 else "down"

    if not is_significant:
        confidence_note = (
            "This model's backtested accuracy was not statistically distinguishable "
            "from a coin flip. Treat this prediction as having no reliable edge -- "
            "state that plainly rather than presenting the direction confidently."
        )
    else:
        confidence_note = (
            f"Backtested accuracy ({backtest_acc:.1%}) was statistically distinguishable "
            f"from chance (baseline {baseline_acc:.1%}), but this is still a modest "
            f"statistical signal, not a guarantee -- present it with appropriate caveats."
        )

    return {
        "prediction": pred_class,
        "prediction_probability_up": float(pred_proba[1]),
        "confidence_note": confidence_note,
        "current_price_usd": float(history_df["GLD"].iloc[-1]),
        "model_accuracy_vs_baseline": {"model": backtest_acc, "baseline": baseline_acc} if backtest_acc else None,
        "is_statistically_significant": is_significant,
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    history_df = fetch_history_from_backend()
    if history_df is None or len(history_df) == 0:
        print("No history on backend yet -- seeding from local CSV (first-ever run).")
        history_df = load_seed_history()

    price, timestamp = fetch_live_price()
    print(f"Fetched live price: ${price:.2f} at {timestamp.isoformat()}")

    live_row = pd.DataFrame([{"Date": pd.Timestamp(timestamp).tz_localize(None), "GLD": price}])
    history_df = pd.concat([history_df, live_row], ignore_index=True)
    history_df = history_df.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)

    result = run_prediction_pipeline(history_df)
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    result["data_points_used"] = len(history_df)

    print("Prediction result:")
    print(json.dumps({k: v for k, v in result.items()}, indent=2))

    push_history_and_prediction(history_df, result)


if __name__ == "__main__":
    main()


