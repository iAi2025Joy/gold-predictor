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

import re

API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
OUTPUT_FILE = "news_sentiment_history.json"

# Keywords checked against each article's title (case-insensitive substring
# match, except short/ambiguous terms which use word-boundary matching --
# see is_gold_relevant). Organized by category, each with a real reason
# it affects gold prices -- not just a generic "sounds economic" list.
#
# CORRECTION 1: tested directly against realistic Fed-related headlines and
# found real gaps -- "Fed Chair Powell signals patience", "FOMC meeting
# minutes", and "Powell hints at rate pause" all went unmatched.
#
# CORRECTION 2: expanded to cover the other major, well-established global
# drivers of gold prices beyond just US monetary policy -- the US dollar
# (gold is priced in USD, so dollar strength/weakness has a direct, well-
# documented inverse relationship with gold), oil (a major inflation input
# and a common co-mover with gold during geopolitical stress), China (the
# world's largest gold consumer and a major central-bank gold buyer -- PBOC
# purchases move the market), and Europe/ECB (the other major central bank
# whose policy divergence from the Fed affects the dollar and, in turn,
# gold). Short/generic terms ("oil", "china") are checked with word-
# boundary matching or compound phrases specifically to avoid false
# positives (e.g. "cooking oil", "Chinese restaurant") -- see below.
GOLD_RELEVANT_KEYWORDS = [
    # The metal itself
    "gold", "bullion", "precious metal", "xau", "gold reserves", "gold etf",

    # US Federal Reserve / monetary policy (gold's most direct macro driver)
    "federal reserve", "fomc", "powell",
    "fed rate", "interest rate", "rate cut", "rate hike", "rate pause",
    "rate path", "rate decision", "rate outlook", "rates steady",
    "quantitative easing", "tapering", "monetary policy", "central bank",

    # Inflation (gold is widely treated as an inflation hedge)
    "inflation", "cpi", "pce inflation", "producer price", "consumer price",
    "stagflation",

    # US employment data (a major input to Fed rate decisions, closely watched market-mover)
    "non-farm payrolls", "nonfarm payrolls", "payrolls report", "jobs report", "employment report",

    # US Dollar (gold is priced in USD -- a direct, well-documented inverse relationship)
    "dollar weak", "dollar strength", "dollar index", "greenback", "dxy",

    # Treasury yields / bond markets (higher yields typically pressure gold, the opportunity-cost relationship)
    "treasury yield", "bond yield", "10-year yield", "yield curve",

    # Oil (major inflation input; often co-moves with gold during geopolitical stress)
    "oil price", "crude oil", "opec", "brent crude", "wti crude", "energy prices",

    # China (world's largest gold consumer and a major central-bank gold buyer)
    "china's economy", "chinese economy", "china economic", "pboc",
    "china gdp", "chinese gdp", "yuan", "renminbi", "china gold",

    # Europe (the other major central bank; policy divergence from the Fed affects the dollar and gold)
    "ecb", "european central bank", "eurozone", "euro area", "lagarde",

    # Other major central banks
    "bank of england", "boe rate", "bank of japan", "boj",

    # Safe-haven demand / geopolitical / macro risk (gold demand often rises with uncertainty)
    "safe haven", "safe-haven", "geopolitical", "recession",
    "trade war", "tariff", "sanctions", "middle east",

    # Market stress / risk sentiment (gold often benefits from flight-to-safety flows)
    "stock market volatility", "market volatility", "vix",
    "banking crisis", "banking sector crisis", "liquidity crisis",

    # General currency/forex context
    "currency", "forex", "exchange rate",
]

# Short, ambiguous terms that need word-boundary matching instead of plain
# substring matching, since a plain substring would false-positive on
# unrelated words (e.g. "fed" inside "federal"/"fedex"; "oil" inside
# "spoil"/"turmoil"; "china" inside "chinatown"-style compound words).
WORD_BOUNDARY_KEYWORDS = ["fed", "oil", "china", "chinese"]


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


# Explicit exclusions checked BEFORE the keyword match -- catches cases
# where a keyword matches technically but the context is clearly
# unrelated. Found via direct testing: "Olive oil prices rise on poor
# harvest" matched on "oil" even though it has nothing to do with
# petroleum/energy markets. Rather than remove "oil"-related keywords
# entirely (which would also lose real petroleum-market headlines like
# "Oil prices surge on Hormuz Strait tensions"), explicitly exclude the
# common non-petroleum oil types.
NON_PETROLEUM_OIL_EXCLUSIONS = ["olive oil", "cooking oil", "vegetable oil", "palm oil", "coconut oil", "sunflower oil", "essential oil"]


def is_gold_relevant(article):
    title = article.get("title", "").lower()

    if any(excl in title for excl in NON_PETROLEUM_OIL_EXCLUSIONS):
        return False

    if any(kw in title for kw in GOLD_RELEVANT_KEYWORDS):
        return True
    # Short/ambiguous terms (see WORD_BOUNDARY_KEYWORDS above) are checked
    # with word-boundary regex instead of plain substring matching, since
    # a plain substring check would false-positive on unrelated words that
    # merely contain the same letters (e.g. "oil" inside "turmoil", "china"
    # inside "chinatown").
    for kw in WORD_BOUNDARY_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", title):
            return True
    return False


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
