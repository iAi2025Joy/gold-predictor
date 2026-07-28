"""
dxy_predictor_updater.py
===========================

US Dollar Index (DXY) counterpart to gold_predictor_updater.py -- same
architecture, same honesty standards (dual-model selection with real
held-out significance tests, recency-weighted training, live rolling
accuracy tracking, direction/price consistency checking).

KEY ARCHITECTURAL DIFFERENCE FROM GOLD/OIL: this script needs NO NEW
DATA SOURCE and NO NEW API KEY. macro_data_fetcher.py already pulls a
real DXY reading (DTWEXBGS, FRED's Trade-Weighted Broad Dollar Index)
every 6 hours into macro_history.json, alongside real_yield_10y and vix
-- this script just trains directly on that already-collected data. This
also means the "price" being predicted here is DTWEXBGS, a real, free,
reputable dollar-strength index -- not the ICE dollar futures ticker
"DXY" itself (which is a different, licensed data product this project
has no free/legal way to access). Framed honestly throughout as "the
Dollar Index (DTWEXBGS)" rather than implying it's the exact same series
professional FX traders call "DXY".

HONEST STARTUP LIMITATION: macro data collection only began recently, so
this will report "insufficient_data" for roughly the first 1-2 weeks
until MIN_ROWS_FOR_PREDICTION worth of real 6-hourly readings accumulate
-- same honest startup behavior gold_predictor_updater.py had on day one.

CADENCE: run every 6 hours, offset AFTER update_macro_data.yml so fresh
macro data has already landed in the repo -- see update_dxy_prediction.yml.
Since the underlying FRED series are themselves daily-resolution, running
more often than every 6 hours would not add real signal.

REUSED, NOT DUPLICATED, INPUTS (documented explicitly for transparency):
- News sentiment: reuses the SAME gold-relevant sentiment score from
  news_sentiment_history.json (Fed/rate/dollar-strength language is
  already heavily represented in that keyword set) rather than building
  a third redundant Alpha Vantage keyword filter for a very similar
  macro-news signal. Labeled honestly as a "macro/Fed-relevant sentiment
  proxy" in the output, not claimed to be USD-exclusive.
- Economic calendar: reuses the SAME economic_calendar.json (FOMC/CPI/NFP
  dates matter directly to the dollar, arguably even more directly than
  to gold).
"""

import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

# Reuse the exact same, already-tested calendar/event-proximity logic
# from the gold script rather than re-implementing it -- see that
# module's add_event_proximity_features docstring for why the forward
# merge_asof is legitimate here (public, pre-announced event dates).
import gold_predictor_updater as gpu

MACRO_FILE = "macro_history.json"
SENTIMENT_FILE = "news_sentiment_history.json"
PREDICTION_FILE = "dxy_prediction_latest.json"
TRACK_RECORD_FILE = "dxy_prediction_track_record.json"
MIN_ROWS_FOR_PREDICTION = 60
HALF_LIFE_DAYS = 30
ROLLING_WINDOW = 50


FED_RATE_OUTLOOK_FILE = "fed_rate_outlook.json"


def load_fed_rate_outlook():
    """Hand-maintained file with the Fed's own real, published Summary of
    Economic Projections (SEP) median rate path -- see fed_rate_outlook.json
    for the honesty note on why this isn't market-implied probability."""
    if not os.path.exists(FED_RATE_OUTLOOK_FILE):
        return None
    with open(FED_RATE_OUTLOOK_FILE) as f:
        return json.load(f)


def get_current_fed_funds_range():
    """Current real Fed funds target range, live from macro_history.json
    (fetched from FRED's DFEDTARU/DFEDTARL by macro_data_fetcher.py)."""
    if not os.path.exists(MACRO_FILE):
        return None, None
    with open(MACRO_FILE) as f:
        records = json.load(f)
    for rec in reversed(records):  # most recent entry with real values
        upper = rec.get("fed_funds_upper")
        lower = rec.get("fed_funds_lower")
        if upper is not None and lower is not None:
            return float(lower), float(upper)
    return None, None


# ============================================================
# 1. BUILD DXY HISTORY FROM ALREADY-COLLECTED MACRO DATA
# ============================================================

def load_dxy_history():
    """Extract a DXY (DTWEXBGS) price-like series directly from
    macro_history.json -- no separate fetch needed. Also carries
    real_yield_10y and vix along as cross-asset features, since they're
    already sitting in the same rows."""
    if not os.path.exists(MACRO_FILE):
        return pd.DataFrame(columns=["Date", "DXY", "real_yield_10y", "vix"])
    with open(MACRO_FILE) as f:
        records = json.load(f)
    if not records:
        return pd.DataFrame(columns=["Date", "DXY", "real_yield_10y", "vix"])

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.rename(columns={"dxy": "DXY"})
    df = df.dropna(subset=["DXY"])
    df = df[["Date", "DXY", "real_yield_10y", "vix"]].sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset="Date", keep="last")
    return df


# ============================================================
# 2. FEATURE ENGINEERING (same formulas as gold, applied to DXY's own series)
# ============================================================

def add_dxy_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["dxy_ret_1d"] = df["DXY"].pct_change(1)
    df["dxy_ret_3d"] = df["DXY"].pct_change(3)
    df["dxy_ret_5d"] = df["DXY"].pct_change(5)
    df["dxy_ma5"] = df["DXY"].rolling(5).mean()
    df["dxy_ma20"] = df["DXY"].rolling(20).mean()
    df["dxy_ma_ratio"] = df["dxy_ma5"] / df["dxy_ma20"]
    df["dxy_vol10"] = df["dxy_ret_1d"].rolling(10).std()
    delta = df["DXY"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["next_ret"] = df["DXY"].shift(-1) / df["DXY"] - 1
    df["target_up"] = (df["next_ret"] > 0).astype(int)
    return df


def merge_macro_sentiment(df):
    """Reuse the same lookahead-safe merge_asof pattern as gold's
    merge_news_sentiment -- documented above as a deliberate reuse of the
    gold-relevant sentiment score, not a new USD-specific filter."""
    df = df.copy()
    if not os.path.exists(SENTIMENT_FILE):
        df["news_sentiment"] = 0.0
        df["news_sentiment_available"] = 0
        return df
    with open(SENTIMENT_FILE) as f:
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
    merged = pd.merge_asof(df, sent_df, left_on="Date", right_on="timestamp", direction="backward")
    merged["news_sentiment_available"] = merged["avg_sentiment_score"].notna().astype(int)
    merged["news_sentiment"] = merged["avg_sentiment_score"].fillna(0.0)
    return merged.drop(columns=["timestamp", "avg_sentiment_score"])


FEATURE_COLS = ["dxy_ret_1d", "dxy_ret_3d", "dxy_ret_5d", "dxy_ma_ratio", "dxy_vol10", "rsi14",
                 "news_sentiment", "news_sentiment_available",
                 "real_yield_10y", "vix",
                 "hours_to_next_event", "in_event_window_48h"]


# ============================================================
# 3. PREDICTION PIPELINE (same dual-model selection as gold)
# ============================================================

def run_prediction_pipeline(dxy_history_df):
    df = add_dxy_features(dxy_history_df)
    df = merge_macro_sentiment(df)
    calendar_df = gpu.load_economic_calendar()
    df = gpu.add_event_proximity_features(df, calendar_df)
    df = df.rename(columns={"next_event_name": "next_event_name", "next_event_type": "next_event_type"})

    df_model = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns] + ["target_up"]).reset_index(drop=True)
    real_yield_available = df["real_yield_10y"].notna()
    vix_available = df["vix"].notna()

    if len(df_model) < MIN_ROWS_FOR_PREDICTION:
        fed_funds_lower, fed_funds_upper = get_current_fed_funds_range()
        fed_rate_outlook = load_fed_rate_outlook()
        return {
            "prediction": "insufficient_data",
            "confidence_note": (
                f"Only {len(df_model)} usable data points; need at least {MIN_ROWS_FOR_PREDICTION}. "
                f"Macro data collection (which this model trains on) only recently began, so this is "
                f"expected to take roughly 1-2 weeks to clear, not an error."
            ),
            "model_type_used": None,
            "current_dxy": float(dxy_history_df["DXY"].iloc[-1]) if len(dxy_history_df) else None,
            "predicted_dxy": None,
            "price_model_type_used": None,
            "price_direction_vs_current": None,
            "direction_price_agreement": None,
            "price_confidence_note": "Not enough data yet to make a forecast.",
            "is_price_prediction_significant": False,
            "model_accuracy_vs_baseline": None,
            "is_statistically_significant": False,
            "current_fed_funds_rate_lower_pct": fed_funds_lower,
            "current_fed_funds_rate_upper_pct": fed_funds_upper,
            "fed_rate_outlook": fed_rate_outlook,
            "fed_rate_outlook_note": (
                "fed_rate_outlook contains the Fed's OWN published median rate projections (the "
                "'dot plot'), NOT market-implied probability -- always attribute correctly. This "
                "field works independently of the DXY prediction model above, so it's available "
                "even while that model is still gathering enough data."
            ),
            "historical_data_start_date": dxy_history_df["Date"].min().strftime("%Y-%m-%d") if len(dxy_history_df) else None,
            "historical_data_end_date": dxy_history_df["Date"].max().strftime("%Y-%m-%d") if len(dxy_history_df) else None,
        }

    split_idx = int(len(df_model) * 0.8)
    train_df, test_df = df_model.iloc[:split_idx], df_model.iloc[split_idx:]
    X_train, y_train = train_df[FEATURE_COLS], train_df["target_up"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target_up"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if len(X_test) else X_train_s[:0]

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
    elif not is_significant:
        confidence_note = (
            f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best on real "
            f"held-out test data) had backtested accuracy ({backtest_acc:.1%}) that was not statistically "
            f"distinguishable from simply always predicting the majority class ({baseline_acc:.1%}) -- "
            f"McNemar's test. Treat this prediction as having no reliable edge beyond the obvious baseline."
        )
    else:
        confidence_note = (
            f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best on real "
            f"held-out test data) had backtested accuracy ({backtest_acc:.1%}) that was statistically "
            f"distinguishable from the majority-class baseline ({baseline_acc:.1%}) via McNemar's test -- "
            f"a modest statistical signal, not a guarantee."
        )

    # --- Price level: LinearRegression vs GradientBoostingRegressor ---
    price_pred_dxy, price_is_significant, price_confidence_note, price_model_type_used = None, False, None, "linear_regression"
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
            f"This forecast ({price_model_type_used.replace('_', ' ')}) had error that was "
            f"{'statistically better than' if price_is_significant else 'NOT statistically better than'} "
            f"simply assuming the index stays the same (Wilcoxon signed-rank test)."
        )

    latest_ret_pred = reg_model.predict(latest_X)[0]
    current_dxy = float(dxy_history_df["DXY"].iloc[-1])
    price_pred_dxy = float(current_dxy * (1 + latest_ret_pred))

    if price_pred_dxy > current_dxy:
        price_direction_vs_current = "higher"
    elif price_pred_dxy < current_dxy:
        price_direction_vs_current = "lower"
    else:
        price_direction_vs_current = "the same"
    direction_price_agreement = (
        (pred_class == "up" and price_pred_dxy >= current_dxy) or
        (pred_class == "down" and price_pred_dxy <= current_dxy)
    )

    latest_news_sentiment = float(latest_row["news_sentiment"].values[0])
    latest_news_sentiment_available = bool(latest_row["news_sentiment_available"].values[0])
    latest_real_yield = latest_row["real_yield_10y"].values[0]
    latest_vix = latest_row["vix"].values[0]
    latest_hours_to_event = float(latest_row["hours_to_next_event"].values[0])
    latest_in_event_window = bool(latest_row["in_event_window_48h"].values[0])
    latest_event_name = latest_row["next_event_name"].values[0] if "next_event_name" in latest_row.columns else None
    latest_event_name = None if pd.isna(latest_event_name) else str(latest_event_name)
    calendar_needs_update = latest_event_name is None

    fed_funds_lower, fed_funds_upper = get_current_fed_funds_range()
    fed_rate_outlook = load_fed_rate_outlook()

    return {
        "prediction": pred_class,
        "prediction_probability_up": float(pred_proba[1]),
        "confidence_note": confidence_note,
        "model_type_used": model_type_used,
        "current_dxy": current_dxy,
        "predicted_dxy": price_pred_dxy,
        "price_model_type_used": price_model_type_used,
        "price_direction_vs_current": price_direction_vs_current,
        "direction_price_agreement": direction_price_agreement,
        "price_confidence_note": price_confidence_note,
        "is_price_prediction_significant": price_is_significant,
        "model_accuracy_vs_baseline": {"model": backtest_acc, "baseline": baseline_acc} if backtest_acc else None,
        "is_statistically_significant": is_significant,
        "current_fed_funds_rate_lower_pct": fed_funds_lower,
        "current_fed_funds_rate_upper_pct": fed_funds_upper,
        "fed_rate_outlook": fed_rate_outlook,
        "fed_rate_outlook_note": (
            "fed_rate_outlook contains the Fed's OWN published median rate projections (the "
            "'dot plot' from their Summary of Economic Projections), NOT market-implied probability "
            "(e.g. CME FedWatch odds) -- no free/legal API exists for that. Always attribute this "
            "correctly as 'the Fed's own projection', never as 'the market expects'."
        ),
        "latest_macro_news_sentiment_score": latest_news_sentiment if latest_news_sentiment_available else None,
        "news_sentiment_note": "This reuses the same macro/Fed-relevant news sentiment score computed for the gold model, not a DXY-exclusive filter.",
        "latest_real_yield_10y_pct": float(latest_real_yield) if pd.notna(latest_real_yield) else None,
        "latest_vix": float(latest_vix) if pd.notna(latest_vix) else None,
        "next_economic_event_name": latest_event_name,
        "hours_until_next_economic_event": None if calendar_needs_update else round(latest_hours_to_event, 1),
        "in_high_impact_event_window_48h": latest_in_event_window,
        "economic_calendar_needs_update": calendar_needs_update,
        "recency_weighting_half_life_days": HALF_LIFE_DAYS,
        "data_source_note": "Tracks DTWEXBGS (FRED's Trade-Weighted Broad Dollar Index), a free, reputable dollar-strength benchmark -- not the identical series to the licensed ICE 'DXY' futures ticker some platforms show, though the two move very closely together.",
        "historical_data_start_date": dxy_history_df["Date"].min().strftime("%Y-%m-%d"),
        "historical_data_end_date": dxy_history_df["Date"].max().strftime("%Y-%m-%d"),
    }


# ============================================================
# 4. LIVE PREDICTION TRACK RECORD (same pattern as gold)
# ============================================================

def load_track_record():
    if os.path.exists(TRACK_RECORD_FILE):
        with open(TRACK_RECORD_FILE) as f:
            return json.load(f)
    return []


def save_track_record(track_record):
    with open(TRACK_RECORD_FILE, "w") as f:
        json.dump(track_record, f, indent=2)


def resolve_pending_predictions(track_record, actual_dxy, actual_timestamp):
    resolved_count = 0
    for entry in track_record:
        if entry.get("resolved"):
            continue
        actual_direction = "up" if actual_dxy > entry["dxy_at_prediction"] else "down"
        entry["resolved"] = True
        entry["resolved_at"] = actual_timestamp.isoformat()
        entry["actual_dxy_at_resolution"] = actual_dxy
        entry["actual_direction"] = actual_direction
        entry["direction_correct"] = (actual_direction == entry["prediction_direction"])
        entry["price_error"] = abs(entry["predicted_dxy"] - actual_dxy) if entry.get("predicted_dxy") is not None else None
        resolved_count += 1
    if resolved_count:
        print(f"Resolved {resolved_count} pending DXY prediction(s).")
    return track_record


def record_new_prediction(track_record, result, current_dxy, timestamp):
    if result.get("prediction") not in ("up", "down"):
        return track_record
    track_record.append({
        "predicted_at": timestamp.isoformat(),
        "dxy_at_prediction": current_dxy,
        "prediction_direction": result["prediction"],
        "predicted_dxy": result.get("predicted_dxy"),
        "resolved": False, "resolved_at": None, "actual_dxy_at_resolution": None,
        "actual_direction": None, "direction_correct": None, "price_error": None,
    })
    return track_record


def compute_rolling_live_accuracy(track_record, window=ROLLING_WINDOW):
    resolved = [e for e in track_record if e.get("resolved")]
    recent = resolved[-window:]
    if len(recent) == 0:
        return {
            "rolling_predictions_tracked": 0, "rolling_direction_accuracy": None,
            "rolling_price_mae": None, "rolling_accuracy_is_significant": False,
            "rolling_accuracy_note": "No resolved live predictions yet -- this builds up over time as the job runs.",
        }
    correct = sum(1 for e in recent if e["direction_correct"])
    total = len(recent)
    accuracy = correct / total
    price_errors = [e["price_error"] for e in recent if e.get("price_error") is not None]
    mae = float(np.mean(price_errors)) if price_errors else None
    is_significant = False
    if total >= 20:
        pvalue = stats.binomtest(correct, total, p=0.5, alternative="two-sided").pvalue
        is_significant = bool(pvalue < 0.05 and accuracy > 0.5)
        note = (f"Over the last {total} resolved live predictions, correct {correct} times ({accuracy:.1%}). "
                + ("Statistically distinguishable from a coin flip." if is_significant else
                   "NOT statistically distinguishable from a coin flip -- no demonstrated live edge yet."))
    else:
        note = f"Only {total} resolved live predictions so far (need 20+ for significance). Raw: {correct}/{total} ({accuracy:.1%})."
    return {
        "rolling_predictions_tracked": total, "rolling_direction_accuracy": accuracy,
        "rolling_price_mae": mae, "rolling_accuracy_is_significant": is_significant,
        "rolling_accuracy_note": note,
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():
    dxy_history_df = load_dxy_history()
    track_record = load_track_record()

    if len(dxy_history_df) == 0:
        print("No macro data available yet -- macro_data_fetcher.py needs to run at least once first.")
        fed_funds_lower, fed_funds_upper = get_current_fed_funds_range()
        result = {
            "prediction": "insufficient_data",
            "confidence_note": "No macro data collected yet.",
            "current_fed_funds_rate_lower_pct": fed_funds_lower,
            "current_fed_funds_rate_upper_pct": fed_funds_upper,
            "fed_rate_outlook": load_fed_rate_outlook(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(PREDICTION_FILE, "w") as f:
            json.dump(result, f, indent=2)
        return

    current_dxy = float(dxy_history_df["DXY"].iloc[-1])
    current_timestamp = dxy_history_df["Date"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)

    track_record = resolve_pending_predictions(track_record, current_dxy, current_timestamp)

    result = run_prediction_pipeline(dxy_history_df)
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    result["data_points_used"] = len(dxy_history_df)

    track_record = record_new_prediction(track_record, result, current_dxy, current_timestamp)
    rolling_stats = compute_rolling_live_accuracy(track_record)
    result.update(rolling_stats)

    print("DXY prediction result:")
    print(json.dumps(result, indent=2))

    save_track_record(track_record)
    with open(PREDICTION_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {PREDICTION_FILE} and {TRACK_RECORD_FILE} locally -- the workflow will commit these back to the repo.")


if __name__ == "__main__":
    main()
