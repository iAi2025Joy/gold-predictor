"""
news_sentiment_fetcher.py
============================

Fetches financial news from Alpha Vantage, filters it down to articles
actually relevant to gold prices (Alpha Vantage's own topic categories
are too broad -- confirmed directly: a real test pull of "economy_macro"
and "financial_markets" articles returned 50/50 unrelated company
earnings reports, zero gold-relevant hits), and produces one aggregate
sentiment score per run.

BUDGET: Alpha Vantage's free tier is 25 requests/day, TOTAL, shared
across all uses of the key. This script uses exactly 1 request per run.
Run on a separate, less-frequent schedule than the hourly price job --
see update_news_sentiment.yml (every 6 hours = 4 requests/day, leaving
comfortable headroom).

HOW RELEVANCE FILTERING WORKS
---------------------------------
Alpha Vantage's `topics` parameter returns a broad mix dominated by
individual company earnings news, not macro/gold-relevant content. So
this script pulls a broad batch (topics=economy_macro,financial_markets,
up to 50 articles per call) and then applies its OWN keyword filter over
each article's title + summary, keeping only articles that actually
mention something gold-price-relevant: the metal itself, monetary policy,
inflation, safe-haven/geopolitical language, or currency moves. This is
a standard, transparent technique for narrowing a general news feed to a
specific topic a source's own categories don't cleanly cover.

HONEST EXPECTATION: gold-specific news is genuinely rare in a general
50-article macro/financial pull -- it is normal and expected for many
runs to find zero matching articles. This is reported honestly (as
"articles_matched": 0), not hidden or padded.
"""

import os
import json
import re
from datetime import datetime, timezone

import requests

API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
OUTPUT_FILE = "news_sentiment_history.json"

# Keywords checked against each article's title + summary (case-insensitive
# substring match). Deliberately broad but still gold-relevant -- covers
# the metal itself, the monetary-policy and inflation angle (gold is
# widely treated as an inflation/rate hedge), and the safe-haven/
# geopolitical angle (gold demand often rises with geopolitical risk).
GOLD_RELEVANT_KEYWORDS = [
    "gold", "bullion", "precious metal", "xau",
    "federal reserve", "fed rate", "interest rate", "rate cut", "rate hike",
    "inflation", "cpi", "monetary policy", "central bank",
    "safe haven", "safe-haven", "geopolitical", "recession",
    "dollar weak", "dollar strength", "currency", "treasury yield",
]


def fetch_news_batch():
    if not API_KEY:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not set.")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": "economy_macro,financial_markets",
        "apikey": API_KEY,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "Information" in data and "feed" not in data:
        # e.g. rate-limit or invalid-key message instead of real data
        raise RuntimeError(f"Alpha Vantage returned an info message instead of data: {data['Information']}")
    return data.get("feed", [])


def is_gold_relevant(article):
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    return any(kw in text for kw in GOLD_RELEVANT_KEYWORDS)


def aggregate_sentiment(articles):
    relevant = [a for a in articles if is_gold_relevant(a)]
    if not relevant:
        return {
            "articles_fetched": len(articles),
            "articles_matched": 0,
            "avg_sentiment_score": None,
            "matched_titles": [],
        }
    scores = [float(a["overall_sentiment_score"]) for a in relevant if "overall_sentiment_score" in a]
    avg_score = sum(scores) / len(scores) if scores else None
    return {
        "articles_fetched": len(articles),
        "articles_matched": len(relevant),
        "avg_sentiment_score": avg_score,
        "matched_titles": [a["title"] for a in relevant][:10],  # capped for file size
    }


def load_history():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_history(history):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(history, f, indent=2)


def main():
    articles = fetch_news_batch()
    result = aggregate_sentiment(articles)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    print(f"Fetched {result['articles_fetched']} articles, "
          f"{result['articles_matched']} gold-relevant.")
    if result["avg_sentiment_score"] is not None:
        print(f"Average sentiment score: {result['avg_sentiment_score']:.4f}")
        print("Matched titles:")
        for t in result["matched_titles"]:
            print(f"  - {t}")
    else:
        print("No gold-relevant articles found in this batch (expected/normal for some runs).")

    history = load_history()
    history.append(result)
    save_history(history)
    print(f"\nSaved to {OUTPUT_FILE} ({len(history)} total entries).")


if __name__ == "__main__":
    main()
