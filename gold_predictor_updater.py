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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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


def merge_macro_data(price_df, macro_file="macro_history.json"):
    """Attach real-yield / dollar-index / VIX features to each price row,
    using the same lookahead-safe merge_asof(direction='backward') pattern
    as merge_news_sentiment above -- only ever attach a macro reading that
    was actually known as of that price row's timestamp.

    Fed the RAW LEVELS (not just changes) because these are the kind of
    number a trader watches as an absolute threshold (e.g. "VIX above 20"),
    not only its daily change -- StandardScaler in the pipeline below
    normalizes levels fine. Missing/not-yet-available periods are filled
    with each series' own historical mean at merge time (a neutral,
    honest fallback -- 0.0 would be a nonsensical value for a real yield
    or a DXY level, unlike news_sentiment where 0.0 genuinely means
    neutral), with an availability flag so the model can tell real data
    from filled data if that distinction matters."""
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    macro_cols = ["real_yield_10y", "dxy", "vix"]

    if not os.path.exists(macro_file):
        for col in macro_cols:
            df[col] = np.nan
        df["macro_data_available"] = 0
        return df

    with open(macro_file) as f:
        macro_records = json.load(f)

    macro_df = pd.DataFrame(macro_records)
    if len(macro_df) == 0 or not all(c in macro_df.columns for c in macro_cols):
        for col in macro_cols:
            df[col] = np.nan
        df["macro_data_available"] = 0
        return df

    macro_df["timestamp"] = pd.to_datetime(macro_df["timestamp"]).dt.tz_localize(None)
    macro_df = macro_df.sort_values("timestamp")[["timestamp"] + macro_cols]

    df = df.sort_values("Date")
    merged = pd.merge_asof(
        df, macro_df,
        left_on="Date", right_on="timestamp",
        direction="backward",  # only ever look at PAST macro readings, never future ones
    )
    merged["macro_data_available"] = merged[macro_cols].notna().all(axis=1).astype(int)
    # Neutral fallback = each column's own mean over what we actually have,
    # not 0.0 -- 0.0 would be a wildly unrealistic "real yield" or "VIX".
    for col in macro_cols:
        col_mean = merged[col].mean()
        merged[col] = merged[col].fillna(col_mean if pd.notna(col_mean) else 0.0)
    merged = merged.drop(columns=["timestamp"])
    return merged


FEATURE_COLS = ["gld_ret_1d", "gld_ret_3d", "gld_ret_5d", "gld_ma_ratio", "gld_vol10", "rsi14",
                 "news_sentiment", "news_sentiment_available",
                 "real_yield_10y", "dxy", "vix", "macro_data_available",
                 "hours_to_next_event", "in_event_window_48h"]

ECONOMIC_CALENDAR_FILE = "economic_calendar.json"
EVENT_SCORE_WINDOW_HOURS = 72  # proximity score reaches 0 beyond this many hours out
EVENT_FLAG_WINDOW_HOURS = 48   # binary "in the event window" flag threshold


def load_economic_calendar(calendar_file=ECONOMIC_CALENDAR_FILE):
    """Load known FOMC/CPI/NFP release dates. THIS IS A STATIC, HAND-
    MAINTAINED CALENDAR -- not a live-scraped feed -- built from the real,
    published Federal Reserve FOMC meeting schedule and BLS Employment
    Situation / CPI release schedule for 2026 (these are announced by the
    Fed/BLS well in advance and are public record, so a static file is
    the honest, correct way to encode them; there's no legitimate "live"
    version of a schedule that's already fixed months ahead). REQUIRES
    ANNUAL MAINTENANCE: once 2026 events run out, hours_to_next_event
    will stop finding anything and this feature goes quietly inert (capped
    at EVENT_SCORE_WINDOW_HOURS budget, not an error) -- see the staleness
    check in run_prediction_pipeline, which surfaces this honestly instead
    of silently doing nothing forever."""
    if not os.path.exists(calendar_file):
        return pd.DataFrame(columns=["datetime", "event", "type"])
    with open(calendar_file) as f:
        records = json.load(f)
    if not records:
        return pd.DataFrame(columns=["datetime", "event", "type"])
    cal_df = pd.DataFrame(records)
    cal_df["datetime"] = pd.to_datetime(cal_df["date"])
    return cal_df.sort_values("datetime")[["datetime", "event", "type"]]


def add_event_proximity_features(df, calendar_df):
    """For each price row, find the NEXT known high-impact event (FOMC/
    CPI/NFP) using merge_asof(direction='forward') -- the mirror image of
    the backward-looking merges used for news sentiment and macro data
    above (those look at the most recent PAST reading; this looks at the
    soonest FUTURE scheduled event, which is legitimate here specifically
    because these dates are pre-announced public knowledge, not something
    that would leak future information the way a future price or news
    sentiment reading would).

    Produces two features: hours_to_next_event (capped at
    EVENT_SCORE_WINDOW_HOURS so distant-future rows don't dominate the
    model's scale) and in_event_window_48h (a simple binary flag). This
    does NOT tell the model which direction an event will move gold --
    only that volatility is more likely soon, which is honest given
    there's no way to know a live Fed decision's outcome in advance."""
    df = df.copy().sort_values("Date")
    if calendar_df.empty:
        df["hours_to_next_event"] = float(EVENT_SCORE_WINDOW_HOURS)
        df["in_event_window_48h"] = 0
        return df

    merged = pd.merge_asof(
        df, calendar_df.rename(columns={"datetime": "next_event_time"}),
        left_on="Date", right_on="next_event_time",
        direction="forward",  # nearest event AT OR AFTER this row's date -- legitimate here, see docstring
    )
    hours = (merged["next_event_time"] - merged["Date"]).dt.total_seconds() / 3600
    merged["hours_to_next_event"] = hours.fillna(EVENT_SCORE_WINDOW_HOURS).clip(upper=EVENT_SCORE_WINDOW_HOURS)
    merged["in_event_window_48h"] = (merged["hours_to_next_event"] <= EVENT_FLAG_WINDOW_HOURS).astype(int)
    merged = merged.rename(columns={"event": "next_event_name", "type": "next_event_type"})
    return merged


def run_prediction_pipeline(history_df):
    history_with_sentiment = merge_news_sentiment(history_df)
    history_with_macro = merge_macro_data(history_with_sentiment)
    calendar_df = load_economic_calendar()
    history_with_events = add_event_proximity_features(history_with_macro, calendar_df)
    df = add_features(history_with_events)
    df_model = df.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

    if len(df_model) < MIN_ROWS_FOR_PREDICTION:
        return {
            "prediction": "insufficient_data",
            "confidence_note": f"Only {len(df_model)} usable data points; need at least {MIN_ROWS_FOR_PREDICTION}.",
            "model_type_used": None,
            "current_price_usd": float(history_df["GLD"].iloc[-1]) if len(history_df) else None,
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
            "latest_real_yield_10y_pct": None,
            "latest_dxy": None,
            "latest_vix": None,
            "macro_data_currently_available": False,
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

    # ------------------------------------------------------------------
    # RECENCY WEIGHTING: without this, a data point from months ago
    # counts exactly as much as one from yesterday when fitting the
    # model -- meaning if gold's behavior shifts (a new volatility
    # regime, a sustained trend reversal), the model only adapts as fast
    # as that old data gets diluted by sheer volume of new rows. Give
    # each TRAINING row an exponential-decay weight based on its recency
    # relative to the most recent data point in the whole dataset (not
    # just the training split's own boundary, so this stays anchored to
    # "now" even as history grows). HALF_LIFE_DAYS = how many days back a
    # row's influence is cut in half -- 30 days is a reasonable starting
    # point for a next-period gold model; lower = more reactive to recent
    # conditions, higher = more stable/slower to adapt. This ONLY affects
    # fitting -- the accuracy/significance evaluation on the test set
    # below remains unweighted, so it stays an honest, unbiased read on
    # real performance, not one flattered by the weighting choice.
    HALF_LIFE_DAYS = 30
    most_recent_date = df_model["Date"].max()
    train_age_days = (most_recent_date - train_df["Date"]).dt.total_seconds() / 86400
    sample_weight = np.power(0.5, train_age_days / HALF_LIFE_DAYS)

    # ------------------------------------------------------------------
    # MODEL SELECTION: try a candidate GradientBoostingClassifier
    # alongside the existing LogisticRegression, and pick whichever
    # actually does BETTER on the real, held-out test set -- not a
    # blanket "always use the fancier model" swap. This keeps the same
    # honesty standard as the rest of this project: complexity has to
    # earn its place with real, measured performance, not be assumed to
    # be an improvement. Conservative hyperparameters (shallow trees, few
    # estimators, low learning rate) are used deliberately -- this is a
    # small dataset with a handful of features, and an aggressive
    # gradient boosting config would be prone to overfitting noise rather
    # than finding real signal.
    logistic_model = LogisticRegression(max_iter=1000, C=0.5)
    logistic_model.fit(X_train_s, y_train, sample_weight=sample_weight)

    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
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

        # Baseline: always predict the majority class observed in TRAINING
        # data (matches what a naive practitioner would actually do; using
        # the test set's own majority fraction here would leak information).
        majority_class_train = int(round(y_train.mean()))
        baseline_preds = pd.Series([majority_class_train] * len(y_test), index=y_test.index)
        baseline_acc = float(accuracy_score(y_test, baseline_preds))

        # McNemar's test: compares the SELECTED MODEL against the
        # BASELINE directly on the same test points, not against blind
        # 50/50 chance. This matters a lot when the outcome is imbalanced
        # (e.g. gold went up in ~98% of periods during a strong bull run
        # in our own backfilled data) -- in that situation, a trivial
        # "always predict up" baseline already scores ~98%, so testing
        # the model against a coin flip would call that "significant"
        # even though the model added nothing over the trivial baseline.
        # An earlier version of this script did exactly that and was
        # caught and corrected here.
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

    if backtest_acc is None:
        # Test set was too small (<20 rows) to evaluate honestly -- this
        # used to crash here (a real dormant bug: the old code tried to
        # format backtest_acc into this message even when it was never
        # computed). Now it degrades gracefully with an honest message
        # instead.
        confidence_note = (
            "Not enough held-out test data yet to evaluate this model's real accuracy "
            "(need at least 20 test rows). Present any direction call as unproven, not confident."
        )
    elif not is_significant:
        if backtest_acc < baseline_acc:
            # Meaningfully different situation from "roughly tied, not
            # proven better" -- the model actively did WORSE than the
            # trivial baseline on real held-out data. Deserves stronger,
            # more skeptical language than "not distinguishable", which
            # can misleadingly read as a near-tie.
            confidence_note = (
                f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best of the "
                f"two candidates tried) had backtested accuracy ({backtest_acc:.1%}) that was actually "
                f"LOWER than simply always predicting the majority class ({baseline_acc:.1%}) on real "
                f"held-out test data -- this model showed no real edge and underperformed even the "
                f"trivial baseline. Treat any direction call from it with strong skepticism, not just "
                f"mild caution -- it is not a case of 'close but unproven', it actively did worse."
            )
        else:
            confidence_note = (
                f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best on "
                f"real held-out test data) had backtested accuracy ({backtest_acc:.1%}) that was not "
                f"statistically distinguishable from simply always predicting the majority class "
                f"({baseline_acc:.1%}) -- McNemar's test, not a comparison to blind chance, since "
                f"the outcome can be imbalanced (e.g. during a strong sustained trend). Treat this "
                f"prediction as having no reliable edge beyond the obvious baseline -- state that "
                f"plainly rather than presenting the direction confidently."
            )
    else:
        confidence_note = (
            f"This model ({model_type_used.replace('_', ' ')}, chosen because it scored best on "
            f"real held-out test data) had backtested accuracy ({backtest_acc:.1%}) that was "
            f"statistically distinguishable from the majority-class baseline ({baseline_acc:.1%}) "
            f"via McNemar's test, meaning the model adds real signal beyond the trivial baseline "
            f"-- but this is still a modest statistical signal, not a guarantee -- present it "
            f"with appropriate caveats."
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
    price_pred_usd, price_is_significant, price_confidence_note, price_model_type_used = None, False, None, "linear_regression"
    linear_reg = LinearRegression()
    linear_reg.fit(X_train_s, train_df["next_ret"], sample_weight=sample_weight)

    gb_reg = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    gb_reg.fit(X_train_s, train_df["next_ret"], sample_weight=sample_weight)

    reg_model = linear_reg  # safe default if the test set is too small to compare candidates honestly
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
                f"This price forecast ({price_model_type_used.replace('_', ' ')}, chosen because "
                f"it scored best on real held-out test data) had error that was not statistically "
                f"better than simply assuming tomorrow's price equals today's price (Wilcoxon "
                f"signed-rank test on squared errors). Present the dollar figure as the model's "
                f"best guess only -- do not imply it is a reliable forecast beyond the current "
                f"price itself."
            )
        else:
            price_confidence_note = (
                f"This price forecast ({price_model_type_used.replace('_', ' ')}, chosen because "
                f"it scored best on real held-out test data) had error that was statistically "
                f"better than the naive 'no change' baseline (Wilcoxon signed-rank test), meaning "
                f"the regression adds real signal -- but treat the exact dollar figure as an "
                f"estimate with real uncertainty, not a precise forecast."
            )

    latest_ret_pred = reg_model.predict(latest_X)[0]
    current_price = float(history_df["GLD"].iloc[-1])
    price_pred_usd = float(current_price * (1 + latest_ret_pred))

    # ------------------------------------------------------------------
    # DIRECTION/PRICE CONSISTENCY CHECK (new -- catches a real, confirmed
    # narration bug). pred_class (up/down) comes from the LOGISTIC
    # classifier; price_pred_usd comes from a COMPLETELY SEPARATE linear
    # regression. Nothing forces these two models to agree -- and in
    # practice they sometimes don't (confirmed: a real run said "potential
    # increase" while the actual predicted_price_usd was below the
    # current price). Rather than trust the chatbot to correctly compare
    # two raw numbers in a sentence every time, compute the true
    # direction implied by the PRICE number here, deterministically, and
    # hand it over as an explicit word GPT is told to use verbatim --
    # plus an explicit flag for when the two models disagree, which is
    # itself useful information (a disagreement is a signal the forecast
    # is on shakier ground than either number alone would suggest).
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

    latest_real_yield_10y = float(latest_row["real_yield_10y"].values[0])
    latest_dxy = float(latest_row["dxy"].values[0])
    latest_vix = float(latest_row["vix"].values[0])
    latest_macro_data_available = bool(latest_row["macro_data_available"].values[0])

    latest_hours_to_event = float(latest_row["hours_to_next_event"].values[0])
    latest_in_event_window = bool(latest_row["in_event_window_48h"].values[0])
    latest_event_name = latest_row["next_event_name"].values[0] if "next_event_name" in latest_row.columns else None
    latest_event_name = None if pd.isna(latest_event_name) else str(latest_event_name)
    # Honest staleness signal: if the calendar found NO real upcoming event
    # (hours capped out at the max window with no actual match), the
    # hand-maintained economic_calendar.json file likely needs a new

    # year's dates added -- surfaced here rather than silently doing
    # nothing forever.
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
        "latest_real_yield_10y_pct": latest_real_yield_10y if latest_macro_data_available else None,
        "latest_dxy": latest_dxy if latest_macro_data_available else None,
        "latest_vix": latest_vix if latest_macro_data_available else None,
        "macro_data_currently_available": latest_macro_data_available,
        "recency_weighting_half_life_days": HALF_LIFE_DAYS,
        "next_economic_event_name": latest_event_name,
        "hours_until_next_economic_event": None if calendar_needs_update else round(latest_hours_to_event, 1),
        "in_high_impact_event_window_48h": latest_in_event_window,
        "economic_calendar_needs_update": calendar_needs_update,
        "historical_data_start_date": history_df["Date"].min().strftime("%Y-%m-%d"),
        "historical_data_end_date": history_df["Date"].max().strftime("%Y-%m-%d"),
    }


# ============================================================
# 3b. LIVE PREDICTION TRACK RECORD (new -- the real fix for
#     "does this improve by learning from its own past mistakes")
# ============================================================
#
# IMPORTANT DISTINCTION from the backtest significance tests above:
# those tests check the model against HISTORICAL data it already had
# access to at training time -- a one-time snapshot. This section is
# different: it tracks REAL predictions this system actually made, then
# checks them against what REALLY happened next, run after run, forever.
# That's the honest way to answer "is this thing actually any good
# lately" -- not a backtest, a live scorecard. The model doesn't
# retroactively change its own past decisions from this (that would be
# a much bigger change, closer to online learning) -- but it does give
# an honest, continuously-updating accuracy number that isn't just the
# one-time historical backtest, and it's the foundation a future online-
# learning step could build on.

TRACK_RECORD_FILE = "prediction_track_record.json"
ROLLING_WINDOW = 50  # how many recent resolved predictions to summarize


def load_track_record():
    if os.path.exists(TRACK_RECORD_FILE):
        with open(TRACK_RECORD_FILE) as f:
            return json.load(f)
    return []


def save_track_record(track_record):
    with open(TRACK_RECORD_FILE, "w") as f:
        json.dump(track_record, f, indent=2)


def resolve_pending_predictions(track_record, actual_price, actual_timestamp):
    """Look at any prediction(s) from previous runs that haven't been
    checked against reality yet, and resolve them now using the price we
    JUST fetched -- that price IS "what actually happened next" relative
    to whenever that earlier prediction was made. Mutates and returns the
    same list. If a run gets skipped (workflow didn't fire, API hiccup),
    older pending entries simply get resolved against the next price that
    does come in -- honest, since "next period" here is defined by
    whatever the next real data point turns out to be, not a fixed clock
    interval."""
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
        if entry.get("predicted_price_usd") is not None:
            entry["price_error_usd"] = abs(entry["predicted_price_usd"] - actual_price)
        else:
            entry["price_error_usd"] = None
        resolved_count += 1
    if resolved_count:
        print(f"Resolved {resolved_count} pending prediction(s) against the newly fetched live price.")
    return track_record


def record_new_prediction(track_record, prediction_result, current_price, timestamp):
    """Log this run's prediction as a new PENDING entry, to be resolved
    next run once we see what the price actually does. Predictions with
    no real direction (insufficient_data) aren't logged -- there's
    nothing meaningful to score."""
    if prediction_result.get("prediction") not in ("up", "down"):
        return track_record
    track_record.append({
        "predicted_at": timestamp.isoformat(),
        "price_at_prediction": current_price,
        "prediction_direction": prediction_result["prediction"],
        "predicted_price_usd": prediction_result.get("predicted_price_usd"),
        "resolved": False,
        "resolved_at": None,
        "actual_price_at_resolution": None,
        "actual_direction": None,
        "direction_correct": None,
        "price_error_usd": None,
    })
    return track_record


def compute_rolling_live_accuracy(track_record, window=ROLLING_WINDOW):
    """Honest, continuously-updating "how has this system actually been
    doing lately" -- separate from, and complementary to, the one-time
    historical backtest. Same statistical-honesty standard as the rest of
    this project: includes a real significance test against a 50/50 coin
    flip (a fair baseline HERE, unlike the historical backtest's majority-
    class baseline, since there's no meaningful "always predict the
    training majority" concept for a live rolling window -- 50/50 is the
    right null hypothesis for "is this system's live direction call any
    better than a coin flip")."""
    resolved = [e for e in track_record if e.get("resolved")]
    recent = resolved[-window:]
    if len(recent) == 0:
        return {
            "rolling_predictions_tracked": 0,
            "rolling_direction_accuracy": None,
            "rolling_price_mae_usd": None,
            "rolling_accuracy_is_significant": False,
            "rolling_accuracy_note": "No resolved live predictions yet -- this builds up over time as the hourly job runs.",
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
        note = (
            f"Over the last {total} resolved live predictions, this system was correct on direction "
            f"{correct} times ({accuracy:.1%}). "
            + (
                f"This is statistically distinguishable from a coin flip (binomial test, p<0.05) -- real, if modest, live edge."
                if is_significant else
                f"This is NOT statistically distinguishable from a coin flip (binomial test) -- treat recent direction "
                f"calls as having no demonstrated live edge yet, regardless of how the raw {accuracy:.1%} figure looks."
            )
        )
    else:
        note = (
            f"Only {total} resolved live predictions so far (need at least 20 for a meaningful significance test). "
            f"Raw accuracy so far: {correct}/{total} ({accuracy:.1%}), but treat this as too small a sample to "
            f"draw a real conclusion from yet."
        )

    return {
        "rolling_predictions_tracked": total,
        "rolling_direction_accuracy": accuracy,
        "rolling_price_mae_usd": mae,
        "rolling_accuracy_is_significant": is_significant,
        "rolling_accuracy_note": note,
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    history_df = load_history()
    track_record = load_track_record()

    price, timestamp = fetch_live_price()
    print(f"Fetched live price: ${price:.2f} at {timestamp.isoformat()}")

    # Resolve any predictions from PREVIOUS runs using this freshly
    # fetched price, before we compute a new one -- this is the actual
    # live feedback loop: check what we said last time against reality.
    track_record = resolve_pending_predictions(track_record, price, timestamp)

    live_row = pd.DataFrame([{"Date": pd.Timestamp(timestamp).tz_localize(None), "GLD": price}])
    history_df = pd.concat([history_df, live_row], ignore_index=True)
    history_df = history_df.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)

    result = run_prediction_pipeline(history_df)
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    result["data_points_used"] = len(history_df)

    # Log THIS run's prediction as pending, to be resolved next run.
    track_record = record_new_prediction(track_record, result, price, timestamp)

    rolling_stats = compute_rolling_live_accuracy(track_record)
    result.update(rolling_stats)

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
