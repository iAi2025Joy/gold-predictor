"""
news_sentiment_fetcher.py
============================

Fetches financial news from Alpha Vantage ONCE per run, then applies TWO
separate keyword-relevance passes over the SAME fetched batch: one for gold
(as before), and one for oil (new). No extra API call for oil -- both
domains' sentiment come from the same single fetch, keeping the total
Alpha Vantage budget usage unchanged (still 1 request/run).

Alpha Vantage's own topic categories are too broad -- confirmed directly: a
real test pull of "economy_macro" and "financial_markets" articles returned
50/50 unrelated company earnings reports, zero gold-relevant hits -- so this
script applies its own keyword filters instead, once per commodity.

BUDGET: Alpha Vantage's free tier is 25 requests/day, TOTAL, shared across
all uses of the key (this script + oil_predictor_updater.py's daily WTI
call). This script still uses exactly 1 request per run regardless of how
many commodities it filters for. Run on a separate, less-frequent schedule
than the price jobs -- see update_news_sentiment.yml (every 6 hours = 4
requests/day, leaving comfortable headroom even with the oil predictor's
+1/day added elsewhere).

HOW RELEVANCE FILTERING WORKS
---------------------------------
This script pulls one broad batch (topics=economy_macro,financial_markets,
up to 50 articles per call) and then applies keyword filters over each
article's title, keeping only articles that actually mention something
relevant to each commodity: monetary policy, inflation, currency moves,
plus commodity-specific drivers (safe-haven demand for gold; OPEC+/supply
dynamics for oil). This is a standard, transparent technique for narrowing
a general news feed to a specific topic a source's own categories don't
cleanly cover.

HONEST EXPECTATION: commodity-specific news is genuinely rare in a general
50-article macro/financial pull -- it is normal and expected for many runs
to find zero matching articles for one or both commodities. This is
reported honestly (as "articles_matched": 0), not hidden or padded.
"""

import os
import json
import re
from datetime import datetime, timezone

import requests

API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
GOLD_OUTPUT_FILE = "news_sentiment_history.json"
OIL_OUTPUT_FILE = "oil_news_sentiment_history.json"

# ============================================================
# GOLD keyword set (unchanged from before)
# ============================================================

# Keywords checked against each article's title (case-insensitive substring
# match, except short/ambiguous terms which use word-boundary matching --
# see is_relevant). Organized by category, each with a real reason it
# affects gold prices -- not just a generic "sounds economic" list.
#
# CORRECTION 1: tested directly against realistic Fed-related headlines and
# found real gaps -- "Fed Chair Powell signals patience", "FOMC meeting
# minutes", and "Powell hints at rate pause" all went unmatched.
#
# CORRECTION 2: expanded to cover the other major, well-established global
# drivers of gold prices beyond just US monetary policy -- the US dollar,
# oil, China, and Europe/ECB. Short/generic terms ("oil", "china") are
# checked with word-boundary matching or compound phrases specifically to
# avoid false positives (e.g. "cooking oil", "Chinese restaurant").
GOLD_RELEVANT_KEYWORDS = [
    # The metal itself
    # NOTE: "gold" itself is intentionally NOT in this plain-substring
    # list -- see WORD_BOUNDARY_KEYWORDS below. Confirmed via real data:
    # a plain substring match on "gold" false-positives on company names
    # like "Goldman Sachs" ("gold" is a literal substring of "goldman").
    "bullion", "precious metal", "xau", "gold reserves", "gold etf",

    # US Federal Reserve / monetary policy (gold's most direct macro driver)
    "federal reserve", "fomc",
    # NOTE: "powell" is intentionally NOT in this plain-substring list --
    # see POWELL_COMPANY_EXCLUSIONS and is_relevant() for why.
    "fed rate", "interest rate", "rate cut", "rate hike", "rate pause",
    "rate path", "rate decision", "rate outlook", "rates steady",
    "quantitative easing", "tapering", "monetary policy", "central bank",

    # Inflation (gold is widely treated as an inflation hedge)
    "inflation", "cpi", "pce inflation", "producer price", "consumer price",
    "stagflation",

    # US employment data (a major input to Fed rate decisions)
    "non-farm payrolls", "nonfarm payrolls", "payrolls report", "jobs report", "employment report",

    # US Dollar (gold is priced in USD -- direct inverse relationship)
    "dollar weak", "dollar strength", "dollar index", "greenback", "dxy",

    # Treasury yields / bond markets (opportunity-cost relationship)
    "treasury yield", "bond yield", "10-year yield", "yield curve",

    # Oil (major inflation input; often co-moves with gold during geopolitical stress)
    "oil price", "crude oil", "opec", "brent crude", "wti crude", "energy prices",

    # China (world's largest gold consumer and a major central-bank gold buyer)
    "china's economy", "chinese economy", "china economic", "pboc",
    "china gdp", "chinese gdp", "yuan", "renminbi", "china gold",

    # Europe (the other major central bank)
    "ecb", "european central bank", "eurozone", "euro area", "lagarde",

    # Other major central banks
    "bank of england", "boe rate", "bank of japan", "boj",

    # Safe-haven demand / geopolitical / macro risk
    "safe haven", "safe-haven", "geopolitical", "recession",
    "trade war", "tariff", "sanctions", "middle east",

    # Market stress / risk sentiment
    "stock market volatility", "market volatility", "vix",
    "banking crisis", "banking sector crisis", "liquidity crisis",

    # General currency/forex context
    "currency", "forex", "exchange rate",
]

GOLD_WORD_BOUNDARY_KEYWORDS = ["fed", "oil", "china", "chinese", "gold", "powell"]

# Explicit exclusions for gold: (1) genuine whole-word name collisions
# word-boundary matching can't distinguish on its own -- "Powell" (Fed
# Chair) vs "Powell Industries" (ticker POWL, unrelated company); and
# (2) non-petroleum "oil" false positives on the word-boundary "oil"
# match -- confirmed via real testing that "Olive oil prices rise on poor
# harvest" was incorrectly matched. Both were confirmed via real data.
GOLD_EXCLUSIONS = [
    "powell industries",
    "olive oil", "cooking oil", "vegetable oil", "palm oil", "coconut oil", "sunflower oil", "essential oil",
]


# ============================================================
# OIL keyword set (new)
# ============================================================

# Oil has real drivers distinct from gold's -- OPEC+ supply decisions,
# inventory reports, refinery/production disruptions, and demand forecasts
# matter enormously for oil specifically and aren't gold-relevant at all.
# Reuses the same monetary-policy/dollar/macro-risk categories where they
# genuinely apply to both (e.g. a recession affects oil demand too), but
# adds oil-specific supply/demand language gold doesn't care about.
OIL_RELEVANT_KEYWORDS = [
    # The commodity itself and its benchmarks
    "crude oil", "wti crude", "brent crude", "crude inventories",
    "oil price", "oil prices", "petroleum", "barrel",

    # OPEC+ and supply-side decisions (the single biggest oil-specific driver)
    "opec+", "opec plus", "production quota", "production cut", "output cut",
    "supply cut", "output increase", "production increase",

    # US inventory/supply data (EIA, API reports -- closely watched weekly market-movers)
    "eia inventory", "eia report", "crude stockpile", "crude stockpiles",
    "api inventory", "strategic petroleum reserve", "spr release",

    # Infrastructure / disruption events
    "refinery outage", "refinery fire", "pipeline disruption", "oil rig",
    "drilling rig", "shale production", "shale output",

    # Geopolitical supply-risk language specific to oil chokepoints/producers
    "strait of hormuz", "strait of hormuz", "oil sanctions", "oil embargo",
    "saudi arabia oil", "russian oil", "oil exports", "venezuela oil",

    # Demand-side macro (shared relevance with gold, but genuinely also oil-specific)
    "oil demand", "fuel demand", "gasoline demand", "diesel demand",
    "global oil demand", "recession", "china oil demand",

    # Currency (oil, like gold, is priced in USD)
    "dollar strength", "dollar weak", "dollar index",
]

OIL_WORD_BOUNDARY_KEYWORDS = ["oil", "china", "opec"]

# Same false-positive class already caught for gold's "oil" keyword --
# "olive oil" etc. are not petroleum-market news.
OIL_EXCLUSIONS = ["olive oil", "cooking oil", "vegetable oil", "palm oil", "coconut oil", "sunflower oil", "essential oil"]


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
        raise RuntimeError(f"Alpha Vantage returned an info message instead of data: {data['Information']}")
    return data.get("feed", [])


def is_relevant(article, keywords, word_boundary_keywords, exclusions):
    """Generalized version of the gold-only is_gold_relevant() -- same
    proven matching logic (exclusions checked first, then plain substring,
    then word-boundary regex for short/ambiguous terms), parameterized so
    both gold and oil reuse the identical, already-tested approach rather
    than duplicating and potentially drifting logic."""
    title = article.get("title", "").lower()

    if any(excl in title for excl in exclusions):
        return False
    if any(kw in title for kw in keywords):
        return True
    for kw in word_boundary_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", title):
            return True
    return False


def aggregate_sentiment(articles, keywords, word_boundary_keywords, exclusions):
    relevant = [a for a in articles if is_relevant(a, keywords, word_boundary_keywords, exclusions)]

    # De-duplicate by title -- confirmed via real data that Alpha Vantage
    # can return the same article twice in one batch, which without this
    # step gets double-counted toward articles_matched and double-weighted
    # in the average sentiment score.
    seen_titles = set()
    deduped = []
    for a in relevant:
        t = a.get("title", "")
        if t not in seen_titles:
            seen_titles.add(t)
            deduped.append(a)
    relevant = deduped

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


def load_history(output_file):
    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)
    return []


def save_history(history, output_file):
    with open(output_file, "w") as f:
        json.dump(history, f, indent=2)


def main():
    articles = fetch_news_batch()

    gold_result = aggregate_sentiment(articles, GOLD_RELEVANT_KEYWORDS, GOLD_WORD_BOUNDARY_KEYWORDS, GOLD_EXCLUSIONS)
    gold_result["timestamp"] = datetime.now(timezone.utc).isoformat()

    oil_result = aggregate_sentiment(articles, OIL_RELEVANT_KEYWORDS, OIL_WORD_BOUNDARY_KEYWORDS, OIL_EXCLUSIONS)
    oil_result["timestamp"] = datetime.now(timezone.utc).isoformat()

    print(f"Fetched {len(articles)} articles total (one shared fetch for both commodities).")
    print(f"\nGOLD: {gold_result['articles_matched']} gold-relevant.")
    if gold_result["avg_sentiment_score"] is not None:
        print(f"  Average sentiment score: {gold_result['avg_sentiment_score']:.4f}")
        for t in gold_result["matched_titles"]:
            print(f"  - {t}")
    else:
        print("  No gold-relevant articles found in this batch (expected/normal for some runs).")

    print(f"\nOIL: {oil_result['articles_matched']} oil-relevant.")
    if oil_result["avg_sentiment_score"] is not None:
        print(f"  Average sentiment score: {oil_result['avg_sentiment_score']:.4f}")
        for t in oil_result["matched_titles"]:
            print(f"  - {t}")
    else:
        print("  No oil-relevant articles found in this batch (expected/normal for some runs).")

    gold_history = load_history(GOLD_OUTPUT_FILE)
    gold_history.append(gold_result)
    save_history(gold_history, GOLD_OUTPUT_FILE)

    oil_history = load_history(OIL_OUTPUT_FILE)
    oil_history.append(oil_result)
    save_history(oil_history, OIL_OUTPUT_FILE)

    print(f"\nSaved to {GOLD_OUTPUT_FILE} ({len(gold_history)} total entries) "
          f"and {OIL_OUTPUT_FILE} ({len(oil_history)} total entries).")


if __name__ == "__main__":
    main()

