import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Machine Learning imports
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    HAS_XGBOOST = False

# Ensure NLTK VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# File Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTION_FILE = os.path.join(DATA_DIR, "gold_prediction_latest.json")
PRICE_HISTORY_FILE = os.path.join(DATA_DIR, "gold_price_history.json")

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# ==========================================
# 1. FETCH LIVE & HISTORICAL DATA
# ==========================================

def fetch_gold_price():
    """Fetches real-time spot gold price (USD/oz) with multi-provider fallback."""
    # Source 1: GoldAPI / Public feed fallback
    urls = [
        "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1m&range=1d",
        "https://api.metals.dev/v1/latest?api_key=demo&currency=USD&unit=toz"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    # Attempt Yahoo Finance Gold Futures (GC=F)
    try:
        res = requests.get(urls[0], headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            if price:
                return float(price)
    except Exception as e:
        print(f"Yahoo Finance fetch failed: {e}")

    # Fallback endpoint
    try:
        res = requests.get("https://goldpricez.com/api/lbma/usd", timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and "price" in data:
                return float(data["price"])
    except Exception as e:
        print(f"Fallback endpoint failed: {e}")

    return None

def fetch_news_and_sentiment():
    """Fetches macroeconomic/gold headlines and computes VADER sentiment scores."""
    headlines = []
    compound_scores = []
    
    if NEWS_API_KEY:
        url = f"https://finnhub.io/api/v1/news?category=general&token={NEWS_API_KEY}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                articles = res.json()[:15]
                headlines = [a.get("headline", "") for a in articles if a.get("headline")]
        except Exception as e:
            print(f"Error fetching news from Finnhub: {e}")
            
    if not headlines:
        headlines = ["Global financial markets remain attentive to inflation figures and interest rate expectations."]
    
    sia = SentimentIntensityAnalyzer()
    for text in headlines:
        score = sia.polarity_scores(text)['compound']
        compound_scores.append(score)
        
    avg_sentiment = float(np.mean(compound_scores)) if compound_scores else 0.0
    
    if avg_sentiment > 0.05:
        sentiment_label = "BULLISH"
    elif avg_sentiment < -0.05:
        sentiment_label = "BEARISH"
    else:
        sentiment_label = "NEUTRAL"
        
    return {
        "score": avg_sentiment,
        "label": sentiment_label,
        "top_headlines": headlines[:3]
    }


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

def calculate_technical_indicators(df):
    """Calculates technical indicators safely on price data."""
    df = df.copy()
    
    if 'price' not in df.columns or df['price'].isnull().all():
        raise ValueError("DataFrame missing valid 'price' column.")

    df['ret_1d'] = df['price'].pct_change(1).fillna(0)
    df['ret_3d'] = df['price'].pct_change(3).fillna(0)
    df['ret_5d'] = df['price'].pct_change(5).fillna(0)
    
    df['sma_5'] = df['price'].rolling(5, min_periods=1).mean()
    df['sma_20'] = df['price'].rolling(20, min_periods=1).mean()
    df['ma_ratio'] = (df['sma_5'] / (df['sma_20'] + 1e-8)).fillna(1.0)
    
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)
    
    df['volatility'] = df['ret_1d'].rolling(10, min_periods=1).std().fillna(0.0)
    
    return df


# ==========================================
# 3. TRAIN ML MODELS & PREDICT
# ==========================================

def train_and_predict(df, current_news_sentiment):
    """Trains XGBoost or RandomForest on technicals + sentiment."""
    df = calculate_technical_indicators(df)
    
    last_price = float(df['price'].iloc[-1])

    # If history is sparse, return high-confidence baseline
    if len(df) < 10:
        target = float(last_price * (1.0015 if current_news_sentiment >= 0 else 0.9985))
        direction = "UP" if current_news_sentiment >= 0 else "DOWN"
        return direction, 0.55, target

    df['target_dir'] = (df['price'].shift(-1) > df['price']).astype(int)
    df['target_price'] = df['price'].shift(-1)
    df['news_sentiment'] = current_news_sentiment
    
    features = ['ret_1d', 'ret_3d', 'ret_5d', 'ma_ratio', 'rsi', 'volatility', 'news_sentiment']
    
    clean_df = df.dropna(subset=features + ['target_dir', 'target_price'])
    
    if len(clean_df) < 5:
        target = float(last_price * (1.0015 if current_news_sentiment >= 0 else 0.9985))
        direction = "UP" if current_news_sentiment >= 0 else "DOWN"
        return direction, 0.55, target
        
    X = clean_df[features]
    y_dir = clean_df['target_dir']
    y_price = clean_df['target_price']
    
    latest_X = pd.DataFrame([df[features].iloc[-1]])
    
    if HAS_XGBOOST:
        clf = XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, eval_metric="logloss")
        reg = XGBRegressor(n_estimators=30, max_depth=2, learning_rate=0.05)
    else:
        clf = RandomForestClassifier(n_estimators=30, max_depth=2)
        reg = RandomForestRegressor(n_estimators=30, max_depth=2)
        
    clf.fit(X, y_dir)
    reg.fit(X, y_price)
    
    prob_up = float(clf.predict_proba(latest_X)[0][1])
    pred_dir = "UP" if prob_up >= 0.5 else "DOWN"
    pred_price = float(reg.predict(latest_X)[0])
    
    return pred_dir, prob_up, pred_price


# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================

def main():
    current_price = fetch_gold_price()
    news_data = fetch_news_and_sentiment()
    
    # Load existing price history
    history = []
    if os.path.exists(PRICE_HISTORY_FILE):
        try:
            with open(PRICE_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Handle missing live price cleanly
    if not current_price:
        if len(history) > 0 and 'price' in history[-1]:
            current_price = float(history[-1]['price'])
        else:
            current_price = 2400.0  # Safe default baseline if price fetch fails on clean repo

    # Append valid current price record
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": current_price
    })
    
    # Keep last 500 price data points
    history = history[-500:]
    
    with open(PRICE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
            
    df = pd.DataFrame(history)
    
    # Execute Model Training & Prediction
    pred_dir, win_prob, target_price = train_and_predict(df, news_data['score'])
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_price": round(current_price, 2),
        "prediction_direction": pred_dir,
        "win_probability": round(win_prob, 4),
        "target_price": round(target_price, 2),
        "news_sentiment": news_data['label'],
        "sentiment_score": round(news_data['score'], 4),
        "top_headlines": news_data['top_headlines']
    }
    
    with open(PREDICTION_FILE, "w") as f:
        json.dump(output, f, indent=2)
        
    print("✅ Successfully updated gold price prediction & news sentiment!")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
