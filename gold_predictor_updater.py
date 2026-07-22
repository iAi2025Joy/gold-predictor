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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ==========================================
# 1. FETCH LIVE MARKET DATA & MACRO INDICATORS
# ==========================================

def fetch_yahoo_price(ticker):
    """Generic helper to fetch regular market price from Yahoo Finance chart endpoint."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            if price is not None:
                return float(price)
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
    return None

def fetch_market_data():
    """Fetches Gold, DXY (Dollar Index), and US 10-Year Bond Yield (^TNX)."""
    gold_price = fetch_yahoo_price("GC=F")
    dxy = fetch_yahoo_price("DX-Y.NYB") or fetch_yahoo_price("DX=F")
    treasury_10y = fetch_yahoo_price("^TNX")

    # Fallback for Gold if Yahoo fails
    if not gold_price:
        try:
            res = requests.get("https://goldpricez.com/api/lbma/usd", timeout=10)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict) and "price" in data:
                    gold_price = float(data["price"])
        except Exception as e:
            print(f"Gold fallback endpoint failed: {e}")

    return {
        "gold": gold_price,
        "dxy": dxy if dxy else 104.0,           # Safe structural fallback
        "us10y": treasury_10y if treasury_10y else 4.20  # Safe structural fallback
    }

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
        headlines = ["Global financial markets remain attentive to US monetary policy, European economic data, and Asian demand trends."]
    
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
    """Calculates technical indicators safely on price and macro columns."""
    df = df.copy()
    
    if 'price' not in df.columns or df['price'].isnull().all():
        raise ValueError("DataFrame missing valid 'price' column.")

    # Price Returns
    df['ret_1d'] = df['price'].pct_change(1).fillna(0)
    df['ret_3d'] = df['price'].pct_change(3).fillna(0)
    df['ret_5d'] = df['price'].pct_change(5).fillna(0)
    
    # Moving Averages & RSI
    df['sma_5'] = df['price'].rolling(5, min_periods=1).mean()
    df['sma_20'] = df['price'].rolling(20, min_periods=1).mean()
    df['ma_ratio'] = (df['sma_5'] / (df['sma_20'] + 1e-8)).fillna(1.0)
    
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)
    
    df['volatility'] = df['ret_1d'].rolling(10, min_periods=1).std().fillna(0.0)

    # Macro Indicators (DXY and US10Y Yields)
    if 'dxy' in df.columns:
        df['dxy_change'] = df['dxy'].pct_change(1).fillna(0)
    else:
        df['dxy_change'] = 0.0

    if 'us10y' in df.columns:
        df['us10y_change'] = df['us10y'].pct_change(1).fillna(0)
    else:
        df['us10y_change'] = 0.0
    
    return df


# ==========================================
# 3. TRAIN ML MODELS & PREDICT
# ==========================================

def train_and_predict(df, current_news_sentiment):
    """Trains XGBoost/RandomForest on technicals, macro variables, and news sentiment."""
    df = calculate_technical_indicators(df)
    
    last_price = float(df['price'].iloc[-1])

    # Feature List
    features = ['ret_1d', 'ret_3d', 'ret_5d', 'ma_ratio', 'rsi', 'volatility', 'dxy_change', 'us10y_change', 'news_sentiment']
    df['news_sentiment'] = current_news_sentiment

    # If history is sparse, return a baseline heuristic estimate
    if len(df) < 10:
        target = float(last_price * (1.0015 if current_news_sentiment >= 0 else 0.9985))
        direction = "UP" if current_news_sentiment >= 0 else "DOWN"
        return direction, 0.55, target

    df['target_dir'] = (df['price'].shift(-1) > df['price']).astype(int)
    df['target_price'] = df['price'].shift(-1)
    
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
        clf = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05, eval_metric="logloss")
        reg = XGBRegressor(n_estimators=40, max_depth=3, learning_rate=0.05)
    else:
        clf = RandomForestClassifier(n_estimators=40, max_depth=3)
        reg = RandomForestRegressor(n_estimators=40, max_depth=3)
        
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
    market_data = fetch_market_data()
    news_data = fetch_news_and_sentiment()
    
    # Load existing price history
    history = []
    if os.path.exists(PRICE_HISTORY_FILE):
        try:
            with open(PRICE_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    current_gold = market_data['gold']
    if not current_gold:
        if len(history) > 0 and 'price' in history[-1]:
            current_gold = float(history[-1]['price'])
        else:
            current_gold = 2400.0

    # Append current record with macro fields
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": current_gold,
        "dxy": market_data['dxy'],
        "us10y": market_data['us10y']
    })
    
    # Maintain last 1000 records
    history = history[-1000:]
    
    with open(PRICE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
            
    df = pd.DataFrame(history)
    
    # Execute Model Training & Prediction
    pred_dir, win_prob, target_price = train_and_predict(df, news_data['score'])
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_price": round(current_gold, 2),
        "prediction_direction": pred_dir,
        "win_probability": round(win_prob, 4),
        "target_price": round(target_price, 2),
        "dxy_index": round(market_data['dxy'], 2),
        "us10y_yield": round(market_data['us10y'], 2),
        "news_sentiment": news_data['label'],
        "sentiment_score": round(news_data['score'], 4),
        "top_headlines": news_data['top_headlines']
    }
    
    with open(PREDICTION_FILE, "w") as f:
        json.dump(output, f, indent=2)
        
    print("✅ Successfully updated gold predictions, macro indicators (DXY/US10Y), and sentiment!")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
