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
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Finnhub API Key stored in GitHub Secrets


# ==========================================
# 1. FETCH LIVE & HISTORICAL DATA
# ==========================================

def fetch_gold_price():
    """Fetches real-time spot gold price (USD/oz)."""
    url = "https://goldpricez.com/api/lbma/usd"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return float(data.get("price", 0.0))
    except Exception as e:
        print(f"Error fetching gold price: {e}")
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
                articles = res.json()[:15]  # Top 15 articles
                headlines = [a.get("headline", "") for a in articles if a.get("headline")]
        except Exception as e:
            print(f"Error fetching news: {e}")
            
    # Fallback to default neutral state if no API key or fetch failed
    if not headlines:
        headlines = ["Market trading in steady range awaiting macroeconomic reports."]
    
    # Compute Sentiment Scores using VADER
    sia = SentimentIntensityAnalyzer()
    for text in headlines:
        score = sia.polarity_scores(text)['compound']
        compound_scores.append(score)
        
    avg_sentiment = float(np.mean(compound_scores)) if compound_scores else 0.0
    
    # Determine sentiment qualitative label
    if avg_sentiment > 0.05:
        sentiment_label = "BULLISH"
    elif avg_sentiment < -0.05:
        sentiment_label = "BEARISH"
    else:
        sentiment_label = "NEUTRAL"
        
    return {
        "score": avg_sentiment,
        "label": sentiment_label,
        "top_headlines": headlines[:3]  # Keep top 3 headlines
    }


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

def calculate_technical_indicators(df):
    """Calculates technical indicators for price data."""
    df = df.copy()
    
    # Returns
    df['ret_1d'] = df['price'].pct_change(1)
    df['ret_3d'] = df['price'].pct_change(3)
    df['ret_5d'] = df['price'].pct_change(5)
    
    # Moving Average Ratios
    df['sma_5'] = df['price'].rolling(5).mean()
    df['sma_20'] = df['price'].rolling(20).mean()
    df['ma_ratio'] = df['sma_5'] / df['sma_20']
    
    # RSI (14)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['volatility'] = df['ret_1d'].rolling(10).std()
    
    return df


# ==========================================
# 3. TRAIN ML MODELS & PREDICT
# ==========================================

def train_and_predict(df, current_news_sentiment):
    """Trains XGBoost (or RandomForest) on technicals + news sentiment."""
    df = calculate_technical_indicators(df)
    
    # Target 1: Direction (1 if next price is higher, else 0)
    df['target_dir'] = (df['price'].shift(-1) > df['price']).astype(int)
    # Target 2: Next Price
    df['target_price'] = df['price'].shift(-1)
    
    # Inject current sentiment as a feature column
    df['news_sentiment'] = current_news_sentiment
    
    features = ['ret_1d', 'ret_3d', 'ret_5d', 'ma_ratio', 'rsi', 'volatility', 'news_sentiment']
    
    # Drop empty rows due to indicator rolling windows
    clean_df = df.dropna(subset=features + ['target_dir', 'target_price'])
    
    if len(clean_df) < 20:
        # Fallback heuristic if dataset history is too small
        last_price = df['price'].iloc[-1]
        return "UP", 0.55, float(last_price * 1.002)
        
    X = clean_df[features]
    y_dir = clean_df['target_dir']
    y_price = clean_df['target_price']
    
    # Latest feature vector for inference
    latest_X = pd.DataFrame([df[features].iloc[-1]])
    
    if HAS_XGBOOST:
        clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, eval_metric="logloss")
        reg = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05)
    else:
        clf = RandomForestClassifier(n_estimators=50, max_depth=3)
        reg = RandomForestRegressor(n_estimators=50, max_depth=3)
        
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
    
    # Load or initialize price history
    if os.path.exists(PRICE_HISTORY_FILE):
        with open(PRICE_HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []
        
    # Append current price to history if retrieved
    if current_price:
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": current_price
        })
        # Keep last 500 records
        history = history[-500:]
        with open(PRICE_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
            
    # Prepare DataFrame for modeling
    if len(history) > 0:
        df = pd.DataFrame(history)
    else:
        # Emergency dummy baseline if history is totally empty
        df = pd.DataFrame([{"price": current_price or 2400.0}])
        
    # Execute Model Training & Prediction
    pred_dir, win_prob, target_price = train_and_predict(df, news_data['score'])
    
    # Assemble Final Output
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_price": current_price or df['price'].iloc[-1],
        "prediction_direction": pred_dir,
        "win_probability": round(win_prob, 4),
        "target_price": round(target_price, 2),
        "news_sentiment": news_data['label'],
        "sentiment_score": round(news_data['score'], 4),
        "top_headlines": news_data['top_headlines']
    }
    
    # Save Output JSON
    with open(PREDICTION_FILE, "w") as f:
        json.dump(output, f, indent=2)
        
    print("✅ Successfully updated gold price prediction & news sentiment!")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
