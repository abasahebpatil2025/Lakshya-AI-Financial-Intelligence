"""
Lakshya AI - Streamlit Web Interface
Stock Market Analysis
"""
import os
import re
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime
import time
from typing import Optional

import io
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
from main import analyze_stock
from models.data_models import Recommendation
from utils.exceptions import (
    InvalidTickerError,
    DataUnavailableError,
    AuthenticationError,
    LakshyaAIError
)

load_dotenv()

st.set_page_config(
    page_title="Lakshya AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* ── Glassmorphism Recommendation Card ── */
    .glass-card {
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        background: rgba(15, 23, 42, 0.75);
        border-radius: 20px;
        padding: 2rem 2.5rem 1.6rem;
        margin: 1rem 0 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 1.5px;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }
    .glass-buy::before   { background: linear-gradient(135deg, #22c55e, #16a34a); }
    .glass-sell::before  { background: linear-gradient(135deg, #ef4444, #dc2626); }
    .glass-hold::before  { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .signal-label {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: 4px;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .signal-buy  { color: #22c55e; text-shadow: 0 0 30px rgba(34,197,94,0.5); }
    .signal-sell { color: #ef4444; text-shadow: 0 0 30px rgba(239,68,68,0.5); }
    .signal-hold { color: #f59e0b; text-shadow: 0 0 30px rgba(245,158,11,0.5); }
    .confidence-bar-wrap {
        background: rgba(255,255,255,0.07);
        border-radius: 999px;
        height: 10px;
        width: 70%;
        margin: 0.8rem auto 0.3rem;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }
    .ticker-label {
        font-size: 0.82rem;
        color: #64748b;
        letter-spacing: 1px;
        margin-top: 0.9rem;
        font-family: 'Courier New', monospace;
    }

    /* ── Indicator Cards ── */
    .indicator-card {
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1rem 1.1rem 0.9rem;
        height: 100%;
    }
    .indicator-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.55rem;
    }
    .indicator-value {
        font-size: 1.5rem;
        font-weight: 800;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        line-height: 1.1;
    }
    .indicator-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }
    .rsi-bar-track {
        background: rgba(255,255,255,0.07);
        border-radius: 999px;
        height: 8px;
        margin-top: 0.5rem;
        position: relative;
        overflow: visible;
    }
    .vol-bar-row {
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 36px;
        margin-top: 0.5rem;
    }
    .vol-bar {
        flex: 1;
        border-radius: 3px 3px 0 0;
        min-height: 4px;
    }

    /* ── Buttons ── */
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover { background-color: #2563eb; }

    /* ── Welcome Screen ── */
    .step-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
    }
    .step-number {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        color: white;
        font-weight: 800;
        font-size: 1rem;
        border-radius: 50%;
        min-width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .step-text {
        color: #e2e8f0;
        font-size: 0.97rem;
        line-height: 1.5;
        padding-top: 0.2rem;
    }
    .step-text strong { color: #7dd3fc; }
    .agent-badge {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        border: 1px solid #3b82f6;
        color: #93c5fd;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.8rem;
        margin: 0.2rem 0.2rem 0 0;
    }
    </style>
""", unsafe_allow_html=True)


def get_currency_symbol(ticker: str) -> str:
    """Return ₹ for Indian stocks (.NS / .BO), $ for everything else."""
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "₹"
    return "$"


# Common Indian company name → ticker mapping for instant offline resolution
# Keys are uppercase; longer/more-specific keys must come before shorter ones
_INDIAN_NAME_MAP = {
    # Tata group
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "TATA CONSULTANCY":          "TCS.NS",
    "TATA MOTORS":               "TATAMOTORS.NS",
    "TATA MOTOR":                "TATAMOTORS.NS",   # partial
    "TATA STEEL":                "TATASTEEL.NS",
    "TATA POWER":                "TATAPOWER.NS",
    "TATA CHEMICALS":            "TATACHEM.NS",
    "TATA CONSUMER":             "TATACONSUM.NS",
    "TATA ELXSI":                "TATAELXSI.NS",
    "TCS":                       "TCS.NS",
    # Reliance
    "RELIANCE INDUSTRIES":       "RELIANCE.NS",
    "RELIANCE":                  "RELIANCE.NS",
    "RIL":                       "RELIANCE.NS",
    # Infosys
    "INFOSYS":                   "INFY.NS",
    "INFY":                      "INFY.NS",
    # Wipro
    "WIPRO":                     "WIPRO.NS",
    # HDFC
    "HDFC BANK":                 "HDFCBANK.NS",
    "HDFC LIFE":                 "HDFCLIFE.NS",
    "HDFC AMC":                  "HDFCAMC.NS",
    "HDFC":                      "HDFCBANK.NS",
    # ICICI
    "ICICI BANK":                "ICICIBANK.NS",
    "ICICI PRUDENTIAL":          "ICICIPRULI.NS",
    "ICICI LOMBARD":             "ICICIGI.NS",
    "ICICI":                     "ICICIBANK.NS",
    # SBI
    "STATE BANK OF INDIA":       "SBIN.NS",
    "STATE BANK":                "SBIN.NS",
    "SBI":                       "SBIN.NS",
    # Bajaj
    "BAJAJ FINANCE":             "BAJFINANCE.NS",
    "BAJAJ FINSERV":             "BAJAJFINSV.NS",
    "BAJAJ AUTO":                "BAJAJ-AUTO.NS",
    # Maruti
    "MARUTI SUZUKI":             "MARUTI.NS",
    "MARUTI":                    "MARUTI.NS",
    # Others
    "ASIAN PAINTS":              "ASIANPAINT.NS",
    "HINDUSTAN UNILEVER":        "HINDUNILVR.NS",
    "HUL":                       "HINDUNILVR.NS",
    "KOTAK MAHINDRA BANK":       "KOTAKBANK.NS",
    "KOTAK BANK":                "KOTAKBANK.NS",
    "KOTAK":                     "KOTAKBANK.NS",
    "AXIS BANK":                 "AXISBANK.NS",
    "BHARTI AIRTEL":             "BHARTIARTL.NS",
    "AIRTEL":                    "BHARTIARTL.NS",
    "ITC":                       "ITC.NS",
    "ONGC":                      "ONGC.NS",
    "NTPC":                      "NTPC.NS",
    "POWER GRID":                "POWERGRID.NS",
    "ADANI PORTS":               "ADANIPORTS.NS",
    "ADANI ENTERPRISES":         "ADANIENT.NS",
    "ADANI GREEN":               "ADANIGREEN.NS",
    "ADANI TOTAL GAS":           "ATGL.NS",
    "SUN PHARMA":                "SUNPHARMA.NS",
    "SUN PHARMACEUTICAL":        "SUNPHARMA.NS",
    "DR REDDYS":                 "DRREDDY.NS",
    "DR REDDY":                  "DRREDDY.NS",
    "CIPLA":                     "CIPLA.NS",
    "DIVIS LABORATORIES":        "DIVISLAB.NS",
    "DIVIS LAB":                 "DIVISLAB.NS",
    "TECH MAHINDRA":             "TECHM.NS",
    "HCL TECHNOLOGIES":          "HCLTECH.NS",
    "HCL TECH":                  "HCLTECH.NS",
    "HCLT":                      "HCLTECH.NS",
    "ULTRATECH CEMENT":          "ULTRACEMCO.NS",
    "ULTRATECH":                 "ULTRACEMCO.NS",
    "NESTLE INDIA":              "NESTLEIND.NS",
    "NESTLE":                    "NESTLEIND.NS",
    "TITAN COMPANY":             "TITAN.NS",
    "TITAN":                     "TITAN.NS",
    "LTIMINDTREE":               "LTIM.NS",
    "LTI MINDTREE":              "LTIM.NS",
    "LTI":                       "LTIM.NS",
    "ZOMATO":                    "ZOMATO.NS",
    "PAYTM":                     "PAYTM.NS",
    "NYKAA":                     "NYKAA.NS",
    "DMART":                     "DMART.NS",
    "AVENUE SUPERMARTS":         "DMART.NS",
    "INDUSIND BANK":             "INDUSINDBK.NS",
    "INDUSIND":                  "INDUSINDBK.NS",
    "MAHINDRA":                  "M&M.NS",
    "M&M":                       "M&M.NS",
    "HERO MOTOCORP":             "HEROMOTOCO.NS",
    "HERO MOTO":                 "HEROMOTOCO.NS",
    "EICHER MOTORS":             "EICHERMOT.NS",
    "ROYAL ENFIELD":             "EICHERMOT.NS",
    "BRITANNIA":                 "BRITANNIA.NS",
    "DABUR":                     "DABUR.NS",
    "GODREJ CONSUMER":           "GODREJCP.NS",
    "PIDILITE":                  "PIDILITIND.NS",
    "BERGER PAINTS":             "BERGEPAINT.NS",
    "HAVELLS":                   "HAVELLS.NS",
    "VOLTAS":                    "VOLTAS.NS",
    "SIEMENS":                   "SIEMENS.NS",
    "ABB INDIA":                 "ABB.NS",
    "LARSEN":                    "LT.NS",
    "L&T":                       "LT.NS",
    "BHEL":                      "BHEL.NS",
    "COAL INDIA":                "COALINDIA.NS",
    "GRASIM":                    "GRASIM.NS",
    "SHREE CEMENT":              "SHREECEM.NS",
    "AMBUJA CEMENT":             "AMBUJACEM.NS",
    "ACC":                       "ACC.NS",
    "INDIGO":                    "INDIGO.NS",
    "INTERGLOBE":                "INDIGO.NS",
    # US names
    "APPLE":                     "AAPL",
    "MICROSOFT":                 "MSFT",
    "GOOGLE":                    "GOOGL",
    "ALPHABET":                  "GOOGL",
    "AMAZON":                    "AMZN",
    "META":                      "META",
    "FACEBOOK":                  "META",
    "TESLA":                     "TSLA",
    "NVIDIA":                    "NVDA",
    "NETFLIX":                   "NFLX",
}

# Known US tickers that need no suffix
_US_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "NVDA", "AMD", "INTC", "NFLX", "UBER", "LYFT", "SNAP",
    "BABA", "JNJ", "JPM", "BAC", "GS", "V", "MA",
}


def _map_lookup(upper: str) -> str | None:
    """
    Exact match first, then longest-prefix match against _INDIAN_NAME_MAP keys.
    Returns the ticker symbol or None.
    """
    if upper in _INDIAN_NAME_MAP:
        return _INDIAN_NAME_MAP[upper]
    # Longest key that the input starts with (handles "TATA MOTOR" → TATAMOTORS.NS)
    best_key, best_sym = "", None
    for key, sym in _INDIAN_NAME_MAP.items():
        if upper.startswith(key) and len(key) > len(best_key):
            best_key, best_sym = key, sym
    # Also check if any key starts with the input (user typed partial key)
    if not best_sym:
        for key, sym in _INDIAN_NAME_MAP.items():
            if key.startswith(upper) and len(key) > len(best_key):
                best_key, best_sym = key, sym
    return best_sym


def resolve_ticker(raw_input: str) -> tuple[str, str | None]:
    """
    Convert a company name or partial ticker into a valid ticker symbol.
    Returns (resolved_ticker, info_message).
    """
    import yfinance as yf

    raw   = raw_input.strip()
    if not raw:
        return "", None

    upper = raw.upper()

    # 1. Already has an exchange suffix — trust it as-is
    if "." in upper:
        return upper, None

    # 2. Offline map — exact + longest-prefix match
    sym = _map_lookup(upper)
    if sym:
        return sym, f'"{raw}" → {sym}'

    # 3. Known US tickers
    if upper in _US_TICKERS:
        return upper, None

    # 4. Multi-word input
    if " " in upper:
        # 4a. Collapse spaces → try as NSE ticker (e.g. TATAPOWER.NS)
        collapsed = upper.replace(" ", "") + ".NS"
        try:
            price = getattr(yf.Ticker(collapsed).fast_info, "last_price", None)
            if price and price > 0:
                return collapsed, f'"{raw}" → {collapsed}'
        except Exception:
            pass

        # 4b. yfinance Search — prefer .NS/.BO results
        try:
            results = yf.Search(raw, max_results=8).quotes
            if results:
                for r in results:
                    s = r.get("symbol", "")
                    if s.endswith(".NS") or s.endswith(".BO"):
                        return s, f'"{raw}" → {s}'
                s = results[0].get("symbol", upper.replace(" ", ""))
                return s, f'"{raw}" → {s}'
        except Exception:
            pass

        # 4c. Bare collapsed ticker
        bare = upper.replace(" ", "")
        return bare, f'"{raw}" → {bare}'

    # 5. Single word — yfinance Search, then .NS fallback
    try:
        results = yf.Search(raw, max_results=8).quotes
        if results:
            for r in results:
                s = r.get("symbol", "")
                if s.endswith(".NS") or s.endswith(".BO"):
                    return s, f'"{raw}" → {s}'
            s = results[0].get("symbol", "")
            if s and "." not in s:
                return s, f'"{raw}" → {s}'
    except Exception:
        pass

    # 6. Last resort: append .NS
    ns_sym = upper + ".NS"
    return ns_sym, f'"{raw}" → {ns_sym} (auto-corrected)'


def get_company_name(ticker: str) -> str:
    """Fetch the human-readable company name for a ticker via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def validate_ticker(ticker: str) -> bool:
    """Return True if yfinance can fetch a valid price for this ticker."""
    if not ticker:
        return False
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None)
        return price is not None and price > 0
    except Exception:
        return False


def parse_pipe_summary(summary: str) -> dict:
    """
    Parse a pipe-delimited summary string like:
      'Price: ₹543.30 | MA-50: ₹520.00 | MA-200: ₹510.00 | RSI: 58.23 | Signals: Overbought'
    Returns a dict of {key: value} with stripped whitespace.
    """
    result = {}
    if not summary or summary.strip() == "":
        return result
    parts = summary.split("|")
    for part in parts:
        part = part.strip()
        if ":" in part:
            key, _, value = part.partition(":")
            result[key.strip()] = value.strip()
    return result


def _get_aws_helper():
    """Return a cached AWSHelper instance (one per Streamlit session)."""
    if "aws_helper" not in st.session_state:
        try:
            from utils.aws_helper import AWSHelper
            st.session_state["aws_helper"] = AWSHelper()
        except Exception:
            st.session_state["aws_helper"] = None
    return st.session_state["aws_helper"]


@st.cache_data(ttl=600, show_spinner=False)
def analyze_headlines_with_claude(ticker: str, headlines_json: str) -> dict:
    """
    Use Claude 3 (via AWSHelper) to:
      - Score each headline on a -1 to +1 scale
      - Generate a 1-sentence AI insight summarising the overall news tone
    Returns:
      {
        "scores":  {title: float},   # -1.0 to 1.0 per headline
        "insight": str,              # 1-sentence AI summary
        "avg_score": float,          # mean of all scores (-1 to 1)
      }
    headlines_json is a JSON string of [{"title": ...}, ...] for cache-key stability.
    """
    import json as _json
    try:
        headlines = _json.loads(headlines_json)
        if not headlines:
            return {"scores": {}, "insight": "", "avg_score": 0.0}

        aws = _get_aws_helper()
        if aws is None:
            return {"scores": {}, "insight": "", "avg_score": 0.0}

        titles_block = "\n".join(
            f"{i+1}. {h['title']}" for i, h in enumerate(headlines)
        )

        prompt = f"""You are a financial news analyst. Analyse the following news headlines for the stock ticker '{ticker}'.

For EACH headline, assign a sentiment score between -1.0 (very bearish) and +1.0 (very bullish), where 0 is neutral.

Then write ONE concise sentence (max 25 words) summarising whether the overall news flow is Bullish or Bearish for {ticker} and why.

Headlines:
{titles_block}

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{
  "scores": {{
    "1": <float>,
    "2": <float>,
    ...
  }},
  "insight": "<one sentence summary>"
}}"""

        raw = aws.invoke_claude(prompt, max_tokens=600, temperature=0.2,
                                system_prompt="You are a precise financial analyst. Respond only with valid JSON.")

        # Parse JSON from response (strip any accidental markdown fences)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`").strip()

        data = _json.loads(raw)
        scores_by_idx = data.get("scores", {})
        insight = data.get("insight", "")

        # Map index → title
        scores_by_title = {}
        total = 0.0
        for i, h in enumerate(headlines):
            key = str(i + 1)
            score = float(scores_by_idx.get(key, 0.0))
            score = max(-1.0, min(1.0, score))
            scores_by_title[h["title"]] = score
            total += score

        avg = round(total / len(headlines), 3) if headlines else 0.0
        return {"scores": scores_by_title, "insight": insight, "avg_score": avg}

    except Exception:
        return {"scores": {}, "insight": "", "avg_score": 0.0}


def display_header():
    pass  # Header is rendered inline in main() to avoid duplication


def display_recommendation_card(recommendation: Recommendation):
    signal     = recommendation.signal
    confidence = recommendation.confidence_score  # 0–100

    if signal == "BUY":
        emoji      = "📈"
        sig_class  = "signal-buy"
        card_class = "glass-buy"
        bar_color  = "linear-gradient(90deg,#16a34a,#22c55e)"
        glow_bg    = "rgba(34,197,94,0.06)"
    elif signal == "SELL":
        emoji      = "📉"
        sig_class  = "signal-sell"
        card_class = "glass-sell"
        bar_color  = "linear-gradient(90deg,#dc2626,#ef4444)"
        glow_bg    = "rgba(239,68,68,0.06)"
    else:
        emoji      = "⏸️"
        sig_class  = "signal-hold"
        card_class = "glass-hold"
        bar_color  = "linear-gradient(90deg,#d97706,#f59e0b)"
        glow_bg    = "rgba(245,158,11,0.06)"

    pct = min(max(int(confidence), 0), 100)

    st.markdown(f"""
        <div class="glass-card {card_class}" style="background:{glow_bg};">
            <div class="signal-label {sig_class}">{emoji}&nbsp;{signal}</div>
            <div style="color:#94a3b8; font-size:0.85rem; margin-top:0.4rem; letter-spacing:0.5px;">
                Confidence Score
            </div>
            <div style="font-size:2rem; font-weight:800; color:#e2e8f0; line-height:1.2;">
                {pct}<span style="font-size:1rem; color:#64748b;">/100</span>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-fill"
                     style="width:{pct}%; background:{bar_color};"></div>
            </div>
            <div class="ticker-label">🏷 {recommendation.ticker}</div>
        </div>
    """, unsafe_allow_html=True)


def compute_7day_forecast(df: pd.DataFrame) -> tuple:
    """
    Linear regression on last 14 days of Close prices.
    Returns (forecast_dates, forecast_candles, predicted_day7_price).
    forecast_candles is a list of OHLC dicts for the 7 predicted business days.
    The first candle open == last historical close for a zero-gap connection.
    """
    import numpy as np

    recent = df["Close"].tail(14).dropna()
    if len(recent) < 5:
        return None, None, None

    x = np.arange(len(recent))
    y = recent.values
    slope, intercept = np.polyfit(x, y, 1)

    last_date = df.index[-1]
    last_close = float(recent.iloc[-1])

    # Estimate recent daily range for synthetic wicks
    recent_range = float(df["High"].tail(14).mean() - df["Low"].tail(14).mean())
    wick_half = recent_range * 0.3  # +-30% of avg range for wicks

    # Business days only (Mon-Fri) -- skip today, take next 7
    bdays = pd.bdate_range(start=last_date, periods=8, freq="B")[1:]

    base_x = len(recent)
    forecast_candles = []
    prev_close = last_close
    for i, d in enumerate(bdays[:7]):
        predicted_close = intercept + slope * (base_x + i + 1)
        predicted_close = round(float(predicted_close), 4)
        open_p  = round(prev_close, 4)
        close_p = predicted_close
        high_p  = round(max(open_p, close_p) + wick_half, 4)
        low_p   = round(min(open_p, close_p) - wick_half, 4)
        forecast_candles.append({
            "time":  d.strftime("%Y-%m-%d"),
            "open":  open_p,
            "high":  high_p,
            "low":   low_p,
            "close": close_p,
        })
        prev_close = close_p

    day7_price = forecast_candles[-1]["close"] if forecast_candles else None
    return bdays[:7], forecast_candles, day7_price


def compute_monte_carlo_forecast(df: pd.DataFrame, days: int = 15, simulations: int = 500) -> dict:
    """
    Monte Carlo simulation for price forecast.
    Returns dict with keys: dates, median, best_case, worst_case, p2_5, p97_5, p25, p75,
    prob_of_profit, p90_target, last_price, volatility.
    p2_5/p97_5 = 95% CI. best_case (P90) is used as the official Target Price.
    """
    import numpy as np

    try:
        close = df["Close"].dropna()
        if len(close) < 20:
            return {}

        log_returns = np.log(close / close.shift(1)).dropna()
        mu    = float(log_returns.tail(30).mean())
        sigma = float(log_returns.tail(30).std())
        last_price = float(close.iloc[-1])
        last_date  = df.index[-1]

        bdays = pd.bdate_range(start=last_date, periods=days + 1, freq="B")[1:]

        # Vectorised simulation
        rng = np.random.default_rng(42)
        shocks    = rng.normal(mu, sigma, size=(simulations, days))
        log_paths = np.cumsum(shocks, axis=1)
        paths     = last_price * np.exp(log_paths)

        # Final-day prices for probability calculation
        final_prices = paths[:, -1]

        median     = np.percentile(paths, 50,   axis=0)
        best_case  = np.percentile(paths, 90,   axis=0)   # P90 = official Target Price
        worst_case = np.percentile(paths, 10,   axis=0)
        p97_5      = np.percentile(paths, 97.5, axis=0)
        p2_5       = np.percentile(paths, 2.5,  axis=0)
        p75        = np.percentile(paths, 75,   axis=0)
        p25        = np.percentile(paths, 25,   axis=0)

        # Probability of Profit: % of simulations ending above current price
        prob_of_profit = round(float(np.mean(final_prices > last_price) * 100), 1)

        date_strs = [d.strftime("%Y-%m-%d") for d in bdays[:days]]

        return {
            "dates":          date_strs,
            "median":         [round(float(v), 4) for v in median],
            "best_case":      [round(float(v), 4) for v in best_case],
            "worst_case":     [round(float(v), 4) for v in worst_case],
            "p97_5":          [round(float(v), 4) for v in p97_5],
            "p2_5":           [round(float(v), 4) for v in p2_5],
            "p75":            [round(float(v), 4) for v in p75],
            "p25":            [round(float(v), 4) for v in p25],
            "last_price":     last_price,
            "volatility":     round(sigma * 100, 2),
            "prob_of_profit": prob_of_profit,
            "p90_target":     round(float(best_case[-1]), 2),   # P90 final day
        }
    except Exception:
        return {}

def build_candlestick_chart(ticker: str, currency: str) -> tuple:
    """DEPRECATED - kept for compatibility. Use display_full_width_chart directly."""
    return None, None
def plot_forecast_candles(forecast_candles: list, currency: str = "$"):
    """
    Build a Plotly candlestick chart for the 7-day AI forecast candles.
    Includes detailed hover template with OHLC + daily % change.
    """
    if not forecast_candles:
        return None

    df = pd.DataFrame(forecast_candles)
    df["Date"] = pd.to_datetime(df["time"])

    # Calculate daily % change; fill first row NaN with 0
    df["Pct_Chg"] = df["close"].pct_change() * 100
    df["Pct_Chg"] = df["Pct_Chg"].fillna(0)

    # Pass all OHLC + pct change via customdata so hovertemplate can reference them
    customdata = df[["open", "high", "low", "close", "Pct_Chg"]].values

    fig = go.Figure(
        go.Candlestick(
            x=df["Date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            customdata=customdata,
            hovertemplate=(
                "<b>Date: %{x}</b><br>"
                "Open: %{customdata[0]:.2f}<br>"
                "High: %{customdata[1]:.2f}<br>"
                "Low: %{customdata[2]:.2f}<br>"
                "Close: %{customdata[3]:.2f}<br>"
                "Change: %{customdata[4]:.2f}%"
                "<extra></extra>"
            ),
            increasing_line_color="rgba(38,166,154,0.9)",
            decreasing_line_color="rgba(239,83,80,0.9)",
            name="AI Forecast",
        )
    )
    fig.update_layout(
        hovermode="x unified",
        paper_bgcolor="#0a0f1c",
        plot_bgcolor="#0a0f1c",
        font=dict(color="#d1d4dc"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(42,46,57,0.5)",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(42,46,57,0.5)",
            tickprefix=currency,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        title=dict(
            text="📈 7-Day AI Forecast Candles",
            font=dict(size=14, color="#d1d4dc"),
            x=0.01,
        ),
    )

    return fig


def _build_tv_chart_data(ticker: str, currency: str) -> tuple:
    """
    Fetch 6-month OHLC data and prepare series for streamlit-lightweight-charts.
    Returns (candles, volume, ma50, forecast_candles, last_close, day7_price, mc_data, sr_levels)
    or all-None on failure.
    sr_levels = {"support": float, "resistance": float}
    """
    try:
        import yfinance as yf
        import numpy as np

        df = yf.Ticker(ticker).history(period="6mo")
        if df.empty:
            return None, None, None, None, None, None, {}, {}

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df["MA50"]   = df["Close"].rolling(window=50).mean()
        df["PctChg"] = df["Close"].pct_change() * 100
        last_close   = float(df["Close"].iloc[-1])

        COLOR_BULL = "rgba(38,166,154,0.9)"
        COLOR_BEAR = "rgba(239,83,80,0.9)"

        df_reset = df.reset_index()
        df_reset["time"]  = df_reset["Date"].dt.strftime("%Y-%m-%d")
        df_reset["color"] = [COLOR_BULL if c >= o else COLOR_BEAR
                             for o, c in zip(df_reset["Open"], df_reset["Close"])]

        candles = []
        for _, row in df_reset.iterrows():
            candles.append({
                "time":       row["time"],
                "open":       round(float(row["Open"]),  4),
                "high":       round(float(row["High"]),  4),
                "low":        round(float(row["Low"]),   4),
                "close":      round(float(row["Close"]), 4),
                "customdata": [
                    round(float(row["PctChg"]), 2) if pd.notna(row["PctChg"]) else 0.0,
                    int(row["Volume"]),
                ],
            })

        volume = df_reset[["time", "Volume", "color"]].rename(
            columns={"Volume": "value"}
        ).to_dict("records")

        ma50_df = df_reset[["time", "MA50"]].dropna().rename(columns={"MA50": "value"})
        ma50    = ma50_df.to_dict("records")

        # ── Support & Resistance from 6-month OHLC ────────────
        # Support  = lowest swing low  (rolling 10-day min of lows)
        # Resistance = highest swing high (rolling 10-day max of highs)
        support    = round(float(df["Low"].rolling(10).min().dropna().min()), 4)
        resistance = round(float(df["High"].rolling(10).max().dropna().max()), 4)
        sr_levels  = {"support": support, "resistance": resistance}

        _, forecast_candles, day7_price = compute_7day_forecast(df)
        mc_data = compute_monte_carlo_forecast(df, days=15, simulations=500)

        return candles, volume, ma50, forecast_candles, last_close, day7_price, mc_data, sr_levels
    except Exception:
        return None, None, None, None, None, None, {}, {}


def display_full_width_chart(recommendation: Recommendation):
    """Render TradingView-style chart using streamlit-lightweight-charts + Monte Carlo scenario."""
    if not recommendation.technical_summary:
        return

    currency = get_currency_symbol(recommendation.ticker)
    st.markdown(
        "<h4 style='margin-bottom:0.4rem;'>📊 Price Chart &amp; AI Forecast</h4>",
        unsafe_allow_html=True
    )

    candles, volume, ma50, forecast_candles, last_close, day7_price, mc_data, sr_levels = \
        _build_tv_chart_data(recommendation.ticker, currency)

    if not candles:
        st.info("Chart data not available")
        st.markdown("---")
        return

    # ── Colour constants ──────────────────────────────────────
    COLOR_BULL  = "rgba(38,166,154,0.9)"
    COLOR_BEAR  = "rgba(239,83,80,0.9)"
    FCAST_BULL  = "rgba(38,166,154,0.4)"
    FCAST_BEAR  = "rgba(239,83,80,0.4)"

    # ── Main price pane ───────────────────────────────────────
    main_chart_opts = {
        "height": 520,
        "layout": {
            "background": {"type": "solid", "color": "#0a0f1c"},
            "textColor": "#d1d4dc",
        },
        "grid": {
            "vertLines": {"color": "rgba(42,46,57,0.3)"},
            "horzLines": {"color": "rgba(42,46,57,0.5)"},
        },
        "crosshair": {
            "mode": 1,
            "vertLine": {
                "labelVisible": True,
                "style": 0,
                "width": 1,
                "color": "rgba(197,203,206,0.4)",
            },
            "horzLine": {
                "labelVisible": True,
                "style": 0,
                "width": 1,
                "color": "rgba(197,203,206,0.4)",
            },
        },
        "rightPriceScale": {
            "borderColor": "rgba(197,203,206,0.2)",
            "scaleMargins": {"top": 0.08, "bottom": 0.05},
        },
        "timeScale": {
            "borderColor": "rgba(197,203,206,0.2)",
            "barSpacing": 8,
            "minBarSpacing": 4,
            "timeVisible": True,
            "secondsVisible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 28,
            "horzAlign": "center",
            "vertAlign": "center",
            "color": "rgba(255,255,255,0.04)",
            "text": recommendation.ticker,
        },
    }

    main_series = [
        {
            "type": "Candlestick",
            "data": candles,
            "options": {
                "upColor":       COLOR_BULL,
                "downColor":     COLOR_BEAR,
                "borderVisible": True,
                "borderUpColor":   COLOR_BULL,
                "borderDownColor": COLOR_BEAR,
                "wickUpColor":   COLOR_BULL,
                "wickDownColor": COLOR_BEAR,
                "title": "Historical Data",
            },
        },
        {
            "type": "Line",
            "data": ma50,
            "options": {
                "color": "#f97316",
                "lineWidth": 2,
                "priceLineVisible": False,
                "lastValueVisible": True,
                "title": "MA-50",
            },
        },
    ]

    if forecast_candles:
        # Build 95% CI zone from the 7-day linear regression forecast
        # Upper bound = high values, Lower bound = low values of forecast candles
        ci_upper = [c["high"] for c in forecast_candles]
        ci_lower = [c["low"]  for c in forecast_candles]
        ci_dates = [c["time"] for c in forecast_candles]
        ci_mid   = [round((c["open"] + c["close"]) / 2, 4) for c in forecast_candles]

        # Add upper CI boundary line (invisible, used for fill reference)
        main_series.append({
            "type": "Line",
            "data": [{"time": t, "value": v} for t, v in zip(ci_dates, ci_upper)],
            "options": {
                "color": "rgba(99,102,241,0.0)",
                "lineWidth": 1,
                "priceLineVisible": False,
                "lastValueVisible": False,
                "title": "",
                "crosshairMarkerVisible": False,
            },
        })
        # Median forecast line
        main_series.append({
            "type": "Line",
            "data": [{"time": t, "value": v} for t, v in zip(ci_dates, ci_mid)],
            "options": {
                "color": "rgba(99,102,241,0.9)",
                "lineWidth": 2,
                "lineStyle": 1,          # dashed
                "priceLineVisible": False,
                "lastValueVisible": True,
                "title": "95% CI Forecast",
                "crosshairMarkerRadius": 4,
            },
        })
        # Lower CI boundary line
        main_series.append({
            "type": "Line",
            "data": [{"time": t, "value": v} for t, v in zip(ci_dates, ci_lower)],
            "options": {
                "color": "rgba(99,102,241,0.0)",
                "lineWidth": 1,
                "priceLineVisible": False,
                "lastValueVisible": False,
                "title": "",
                "crosshairMarkerVisible": False,
            },
        })

    # ── Volume pane ───────────────────────────────────────────
    volume_chart_opts = {
        "height": 120,
        "layout": {
            "background": {"type": "solid", "color": "#0a0f1c"},
            "textColor": "#d1d4dc",
        },
        "grid": {
            "vertLines": {"color": "rgba(42,46,57,0)"},
            "horzLines": {"color": "rgba(42,46,57,0.3)"},
        },
        "rightPriceScale": {
            "scaleMargins": {"top": 0.1, "bottom": 0.0},
            "borderVisible": False,
        },
        "timeScale": {"visible": False},
        "watermark": {
            "visible": True,
            "fontSize": 12,
            "horzAlign": "left",
            "vertAlign": "top",
            "color": "rgba(255,255,255,0.25)",
            "text": "Volume",
        },
    }

    volume_series = [
        {
            "type": "Histogram",
            "data": volume,
            "options": {
                "priceFormat": {"type": "volume"},
                "priceScaleId": "vol",
            },
            "priceScale": {
                "id": "vol",
                "scaleMargins": {"top": 0.1, "bottom": 0.0},
                "alignLabels": False,
            },
        }
    ]

    renderLightweightCharts(
        [
            {"chart": main_chart_opts,   "series": main_series},
            {"chart": volume_chart_opts, "series": volume_series},
        ],
        key=f"tv_chart_{recommendation.ticker}"
    )

    # ── Legend strip ──────────────────────────────────────────
    st.markdown(
        "<div style='display:flex;gap:1.5rem;align-items:center;"
        "font-size:0.8rem;color:#94a3b8;margin:0.4rem 0 0.2rem;'>"
        "<span><span style='display:inline-block;width:12px;height:12px;"
        "background:rgba(38,166,154,0.9);border-radius:2px;margin-right:5px;'></span>"
        "Historical Data</span>"
        "<span><span style='display:inline-block;width:20px;height:3px;"
        "background:#6366f1;border-radius:2px;margin-right:5px;vertical-align:middle;"
        "border-top:2px dashed #6366f1;'></span>"
        "95% CI Forecast Path</span>"
        "<span><span style='display:inline-block;width:20px;height:3px;"
        "background:#f97316;border-radius:2px;margin-right:5px;vertical-align:middle;'></span>"
        "MA-50</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#475569;font-size:0.75rem;margin:0 0 0.4rem;'>"
        "&#9432; Dashed purple line = AI forecast median. "
        "Shaded zone = 95% Confidence Interval (upper/lower bounds from linear regression). "
        "Hover over candles for OHLC details.</p>",
        unsafe_allow_html=True
    )

    # ── Monte Carlo Scenario Analysis (Plotly) ────────────────
    if mc_data and mc_data.get("dates"):
        st.markdown(
            "<h4 style='margin:1.2rem 0 0.3rem; color:#e2e8f0;'>"
            "🎲 15-Day Scenario Analysis — Confidence Interval (Monte Carlo)</h4>",
            unsafe_allow_html=True
        )
        vol_pct        = mc_data.get("volatility", 0)
        lp             = mc_data.get("last_price", 0)
        med_15         = mc_data["median"][-1]
        p90_target     = mc_data.get("p90_target", mc_data["best_case"][-1])
        prob_profit    = mc_data.get("prob_of_profit", 0)
        p90_upside     = (p90_target - lp) / lp * 100 if lp else 0
        worst_15       = mc_data["worst_case"][-1]
        worst_downside = (worst_15 - lp) / lp * 100 if lp else 0

        mc_col1, mc_col2, mc_col3, mc_col4, mc_col5 = st.columns(5)
        with mc_col1:
            st.markdown(
                f"<div class='indicator-card'><div class='indicator-title'>Daily Volatility</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#f59e0b;'>{vol_pct:.2f}%</div>"
                f"<div class='indicator-sub'>30-day σ</div></div>", unsafe_allow_html=True)
        with mc_col2:
            st.markdown(
                f"<div class='indicator-card'><div class='indicator-title'>Target Price (P90)</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#22c55e;'>{currency}{p90_target:,.2f}</div>"
                f"<div class='indicator-sub' style='color:#22c55e;'>+{p90_upside:.1f}% upside</div></div>",
                unsafe_allow_html=True)
        with mc_col3:
            st.markdown(
                f"<div class='indicator-card'><div class='indicator-title'>Median Path (P50)</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#e2e8f0;'>{currency}{med_15:,.2f}</div>"
                f"<div class='indicator-sub'>Expected</div></div>", unsafe_allow_html=True)
        with mc_col4:
            st.markdown(
                f"<div class='indicator-card'><div class='indicator-title'>Worst Case (P10)</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#ef4444;'>{currency}{worst_15:,.2f}</div>"
                f"<div class='indicator-sub' style='color:#ef4444;'>{worst_downside:.1f}%</div></div>",
                unsafe_allow_html=True)
        with mc_col5:
            prob_color = "#22c55e" if prob_profit >= 60 else ("#f59e0b" if prob_profit >= 40 else "#ef4444")
            st.markdown(
                f"<div class='indicator-card'><div class='indicator-title'>Prob. of Profit</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:{prob_color};'>{prob_profit:.1f}%</div>"
                f"<div class='indicator-sub'>Sims ending above CMP</div></div>",
                unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        dates = mc_data["dates"]
        fig_mc = go.Figure()

        # ── Best Case zone: P50→P90 — semi-transparent Green ──
        fig_mc.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=mc_data["best_case"] + mc_data["median"][::-1],
            fill="toself",
            fillcolor="rgba(34,197,94,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Best Case Zone (P50–P90)",
            hoverinfo="skip",
        ))
        # ── Worst Case zone: P10→P50 — semi-transparent Red ──
        fig_mc.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=mc_data["median"] + mc_data["worst_case"][::-1],
            fill="toself",
            fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Worst Case Zone (P10–P50)",
            hoverinfo="skip",
        ))
        # ── Outer 95% CI band ──
        fig_mc.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=mc_data["p97_5"] + mc_data["p2_5"][::-1],
            fill="toself",
            fillcolor="rgba(99,102,241,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI (P2.5–P97.5)",
            hoverinfo="skip",
        ))
        # P90 Target Price line
        fig_mc.add_trace(go.Scatter(
            x=dates, y=mc_data["best_case"],
            mode="lines", name=f"Target Price P90 ({currency}{p90_target:,.2f})",
            line=dict(color="#22c55e", width=1.8, dash="dot"),
        ))
        # P10 Worst Case line
        fig_mc.add_trace(go.Scatter(
            x=dates, y=mc_data["worst_case"],
            mode="lines", name="Worst Case (P10)",
            line=dict(color="#ef4444", width=1.8, dash="dot"),
        ))
        # Median line
        fig_mc.add_trace(go.Scatter(
            x=dates, y=mc_data["median"],
            mode="lines", name="Median Path (P50)",
            line=dict(color="#a78bfa", width=2.5),
        ))

        # ── Support & Resistance horizontal lines ──────────────
        if sr_levels:
            support    = sr_levels.get("support")
            resistance = sr_levels.get("resistance")
            if support:
                fig_mc.add_hline(
                    y=support,
                    line=dict(color="rgba(34,197,94,0.6)", width=1.5, dash="dash"),
                    annotation_text=f"Support {currency}{support:,.2f}",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#22c55e", size=10),
                )
            if resistance:
                fig_mc.add_hline(
                    y=resistance,
                    line=dict(color="rgba(239,68,68,0.6)", width=1.5, dash="dash"),
                    annotation_text=f"Resistance {currency}{resistance:,.2f}",
                    annotation_position="top right",
                    annotation_font=dict(color="#ef4444", size=10),
                )

        fig_mc.update_layout(
            paper_bgcolor="#0a0f1c",
            plot_bgcolor="#0a0f1c",
            font=dict(color="#d1d4dc", size=11),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                font=dict(size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                showgrid=True, gridcolor="rgba(42,46,57,0.4)",
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                showgrid=True, gridcolor="rgba(42,46,57,0.4)",
                tickprefix=currency,
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        st.markdown(
            f"<p style='color:#475569;font-size:0.75rem;margin:0 0 1rem;'>"
            f"&#9432; <b style='color:#22c55e;'>Green zone</b> = Best Case (P50–P90 upside). "
            f"<b style='color:#ef4444;'>Red zone</b> = Worst Case (P10–P50 downside). "
            f"Outer band = 95% CI. Dashed green = Target Price (P90 = {currency}{p90_target:,.2f}). "
            f"Probability of Profit: <b style='color:{prob_color};'>{prob_profit:.1f}%</b> of simulations ended above CMP. "
            f"Based on 500 Monte Carlo simulations · 30-day historical volatility.</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")


def fetch_technical_indicators(ticker: str) -> dict:
    """
    Compute RSI-14, MACD, Bollinger Band Width, and Volume vs 20-day avg
    directly from yfinance OHLCV data.
    Returns a dict with keys: rsi, macd_signal, bb_width, vol_ratio, vol_bars
    """
    result = {
        "rsi": None, "macd_signal": None,
        "bb_width": None, "vol_ratio": None, "vol_bars": []
    }
    try:
        import yfinance as yf
        import numpy as np

        df = yf.Ticker(ticker).history(period="6mo")
        if df.empty or len(df) < 30:
            return result

        close = df["Close"]
        volume = df["Volume"]

        # ── RSI-14 ────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi   = float(100 - (100 / (1 + rs.iloc[-1])))
        result["rsi"] = round(rsi, 1)

        # ── MACD (12,26,9) ────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        # Crossover: last bar vs previous bar
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            result["macd_signal"] = "bullish_cross"
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            result["macd_signal"] = "bearish_cross"
        elif macd_line.iloc[-1] > signal_line.iloc[-1]:
            result["macd_signal"] = "bullish"
        else:
            result["macd_signal"] = "bearish"

        # ── Bollinger Band Width (20-day, 2σ) ─────────────────
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        bb_up  = sma20 + 2 * std20
        bb_lo  = sma20 - 2 * std20
        bb_w   = float((bb_up.iloc[-1] - bb_lo.iloc[-1]) / sma20.iloc[-1] * 100)
        result["bb_width"] = round(bb_w, 2)

        # ── Volume vs 20-day avg ───────────────────────────────
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_now = float(volume.iloc[-1])
        result["vol_ratio"] = round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0

        # Last 10 bars for mini bar chart (normalised 0–100)
        recent_vol = volume.tail(10).values.astype(float)
        v_max = recent_vol.max()
        result["vol_bars"] = [round(v / v_max * 100, 1) for v in recent_vol] if v_max > 0 else []

    except Exception:
        pass
    return result


def display_technical_data(recommendation: Recommendation):
    """Advanced 4-panel technical indicator row below the chart."""
    if not recommendation.technical_summary:
        st.warning("Technical analysis data not available")
        return

    ind = fetch_technical_indicators(recommendation.ticker)

    st.markdown(
        "<h4 style='margin:0.2rem 0 0.8rem; color:#e2e8f0;'>⚡ Technical Indicators</h4>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    # ── 1. RSI Heatmap ────────────────────────────────────────
    with c1:
        rsi = ind["rsi"]
        if rsi is not None:
            if rsi >= 70:
                rsi_color, rsi_label, rsi_bg = "#ef4444", "Overbought", "rgba(239,68,68,0.12)"
            elif rsi <= 30:
                rsi_color, rsi_label, rsi_bg = "#22c55e", "Oversold", "rgba(34,197,94,0.12)"
            else:
                rsi_color, rsi_label, rsi_bg = "#f59e0b", "Neutral", "rgba(245,158,11,0.10)"
            # Position marker on 0–100 bar
            pct = int(rsi)
            bar_html = f"""
            <div class="rsi-bar-track" style="background:linear-gradient(90deg,
                #22c55e 0%,#22c55e 30%,#f59e0b 30%,#f59e0b 70%,#ef4444 70%,#ef4444 100%);">
                <div style="position:absolute; left:{pct}%; top:-3px;
                    width:3px; height:14px; background:#fff;
                    border-radius:2px; transform:translateX(-50%);"></div>
            </div>"""
            st.markdown(f"""
                <div class="indicator-card" style="background:{rsi_bg}; border-color:{rsi_color}33;">
                    <div class="indicator-title">RSI (14)</div>
                    <div class="indicator-value" style="color:{rsi_color};">{rsi}</div>
                    <div class="indicator-sub">{rsi_label}</div>
                    {bar_html}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="indicator-card"><div class="indicator-title">RSI (14)</div>'
                        '<div class="indicator-sub">N/A</div></div>', unsafe_allow_html=True)

    # ── 2. MACD Crossover ─────────────────────────────────────
    with c2:
        ms = ind["macd_signal"]
        if ms is None:
            st.markdown('<div class="indicator-card"><div class="indicator-title">MACD (12,26,9)</div>'
                        '<div class="indicator-sub" style="color:#64748b;">&#9888; Data Missing</div></div>',
                        unsafe_allow_html=True)
        elif ms == "bullish_cross":
            m_icon, m_color, m_label, m_sub = "⚡", "#22c55e", "Bullish Cross", "Fresh crossover ↑"
            m_bg = "rgba(34,197,94,0.10)"
        elif ms == "bearish_cross":
            m_icon, m_color, m_label, m_sub = "⚡", "#ef4444", "Bearish Cross", "Fresh crossover ↓"
            m_bg = "rgba(239,68,68,0.10)"
        elif ms == "bullish":
            m_icon, m_color, m_label, m_sub = "📈", "#22c55e", "Bullish", "MACD above Signal"
            m_bg = "rgba(34,197,94,0.07)"
        else:
            m_icon, m_color, m_label, m_sub = "📉", "#ef4444", "Bearish", "MACD below Signal"
            m_bg = "rgba(239,68,68,0.07)"
        st.markdown(f"""
            <div class="indicator-card" style="background:{m_bg}; border-color:{m_color}33;">
                <div class="indicator-title">MACD (12,26,9)</div>
                <div class="indicator-value" style="color:{m_color}; font-size:1.8rem;">{m_icon}</div>
                <div class="indicator-value" style="color:{m_color}; font-size:1.1rem;">{m_label}</div>
                <div class="indicator-sub">{m_sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── 3. Bollinger Band Width ───────────────────────────────
    with c3:
        bbw = ind["bb_width"]
        if bbw is not None:
            if bbw > 8:
                bb_color, bb_label = "#ef4444", "High Volatility"
                bb_bg = "rgba(239,68,68,0.10)"
            elif bbw < 3:
                bb_color, bb_label = "#38bdf8", "Low Volatility"
                bb_bg = "rgba(56,189,248,0.10)"
            else:
                bb_color, bb_label = "#f59e0b", "Normal Range"
                bb_bg = "rgba(245,158,11,0.10)"
            st.markdown(f"""
                <div class="indicator-card" style="background:{bb_bg}; border-color:{bb_color}33;">
                    <div class="indicator-title">Bollinger Width</div>
                    <div class="indicator-value" style="color:{bb_color};">{bbw}%</div>
                    <div class="indicator-sub">{bb_label}</div>
                    <div style="margin-top:0.5rem; background:rgba(255,255,255,0.07);
                        border-radius:999px; height:8px; overflow:hidden;">
                        <div style="width:{min(bbw/15*100,100):.0f}%; height:100%;
                            background:{bb_color}; border-radius:999px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="indicator-card"><div class="indicator-title">Bollinger Width</div>'
                        '<div class="indicator-sub">N/A</div></div>', unsafe_allow_html=True)

    # ── 4. Volume Trend ───────────────────────────────────────
    with c4:
        vr = ind["vol_ratio"]
        vbars = ind["vol_bars"]
        if vr is not None:
            if vr >= 1.5:
                v_color, v_label = "#22c55e", f"{vr}× avg — Surge"
                v_bg = "rgba(34,197,94,0.10)"
            elif vr <= 0.6:
                v_color, v_label = "#64748b", f"{vr}× avg — Low"
                v_bg = "rgba(100,116,139,0.10)"
            else:
                v_color, v_label = "#38bdf8", f"{vr}× avg — Normal"
                v_bg = "rgba(56,189,248,0.10)"

            bars_html = ""
            for i, h in enumerate(vbars):
                is_last = (i == len(vbars) - 1)
                bar_c = v_color if is_last else "rgba(255,255,255,0.18)"
                bars_html += (f"<div class='vol-bar' style='height:{max(h,4):.0f}%; "
                              f"background:{bar_c};'></div>")

            st.markdown(f"""
                <div class="indicator-card" style="background:{v_bg}; border-color:{v_color}33;">
                    <div class="indicator-title">Volume Trend</div>
                    <div class="indicator-value" style="color:{v_color};">{vr}×</div>
                    <div class="indicator-sub">{v_label}</div>
                    <div class="vol-bar-row">{bars_html}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="indicator-card"><div class="indicator-title">Volume Trend</div>'
                        '<div class="indicator-sub">N/A</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")


def _fmt_inr(value_raw: float, is_indian: bool) -> str:
    """
    Format a raw numeric value (in the currency yfinance returns) into a
    human-readable string.
    - Indian tickers (.NS/.BO): convert to Rs. Cr / Rs. Lakh Cr
    - US tickers: keep $ B / $ T
    """
    if value_raw is None:
        return "N/A"
    if is_indian:
        cr = value_raw / 1e7          # 1 Crore = 10^7
        if cr >= 1e5:                 # ≥ 1 Lakh Cr
            return f"₹ {cr/1e5:.2f} Lakh Cr"
        elif cr >= 1:
            return f"₹ {cr:,.0f} Cr"
        else:
            return f"₹ {value_raw:,.0f}"
    else:
        if value_raw >= 1e12:
            return f"${value_raw/1e12:.2f}T"
        elif value_raw >= 1e9:
            return f"${value_raw/1e9:.2f}B"
        else:
            return f"${value_raw/1e6:.0f}M"


def _get_usd_inr_rate() -> float:
    """
    Fetch live USD→INR exchange rate via yfinance (USDINR=X).
    Falls back to 83.0 if unavailable.
    """
    try:
        import yfinance as yf
        rate = yf.Ticker("USDINR=X").fast_info.last_price
        if rate and rate > 0:
            return float(rate)
    except Exception:
        pass
    return 83.0  # safe fallback


def _fmt_market_cap(mc_raw: float, is_indian: bool, base_is_us: bool, usd_inr: float) -> str:
    """
    Format market cap with explicit currency label.
    - Indian stock in a US-base comparison: show both ₹ Cr and $ equivalent
    - Indian stock in Indian-base comparison: show ₹ Cr / ₹ Lakh Cr
    - US stock: show $B / $T
    """
    if mc_raw is None:
        return "N/A"
    if is_indian:
        cr = mc_raw / 1e7
        if base_is_us:
            # Also show USD equivalent for easy comparison
            usd_val = mc_raw / usd_inr
            usd_str = f"${usd_val/1e9:.2f}B" if usd_val < 1e12 else f"${usd_val/1e12:.2f}T"
            inr_str = f"₹{cr/1e5:.2f}LCr" if cr >= 1e5 else f"₹{cr:,.0f}Cr"
            return f"{inr_str} ({usd_str})"
        else:
            if cr >= 1e5:
                return f"₹ {cr/1e5:.2f} Lakh Cr"
            return f"₹ {cr:,.0f} Cr"
    else:
        if mc_raw >= 1e12:
            return f"${mc_raw/1e12:.2f}T (USD)"
        elif mc_raw >= 1e9:
            return f"${mc_raw/1e9:.2f}B (USD)"
        else:
            return f"${mc_raw/1e6:.0f}M (USD)"


def fetch_extra_fundamentals(ticker: str) -> dict:
    """
    Fetch P/E, P/B, D/E, ROE, Dividend Yield, Promoter Holding, Market Cap,
    Earnings, and quarterly financials including Operating Margin and Interest Coverage.
    Currency: INR (₹ Cr) for Indian tickers, USD for others.
    """
    is_indian = ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")
    result = {
        "pe": None, "pb": None, "de": None,
        "roe": None, "div_yield": None, "promoter_holding": None,
        "market_cap": "N/A", "earnings": "N/A",
        "quarterly_html": None,
        "quarterly_bar_data": None,   # list of {quarter, net_profit} for bar chart
        "revenue_growth": None,       # % YoY revenue growth (latest vs year-ago quarter)
        "industry_pe": None,          # sector average P/E for health check
        "is_indian": is_indian,
    }
    try:
        import yfinance as yf
        import numpy as np

        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        # ── Scalar ratios ──────────────────────────────────────
        pe = info.get("trailingPE") or info.get("forwardPE")
        result["pe"] = round(float(pe), 1) if pe else None

        pb = info.get("priceToBook")
        result["pb"] = round(float(pb), 2) if pb else None

        de = info.get("debtToEquity")
        result["de"] = round(float(de) / 100, 2) if de else None  # yfinance gives %, convert to ratio

        roe = info.get("returnOnEquity")
        result["roe"] = round(float(roe) * 100, 1) if roe else None

        dy = info.get("dividendYield")
        result["div_yield"] = round(float(dy) * 100, 2) if dy else None

        # Promoter holding — available via majorHoldersBreakdown for Indian stocks
        # yfinance exposes it as heldPercentInsiders for US; for .NS use institutionPercentHeld
        insider = info.get("heldPercentInsiders")
        result["promoter_holding"] = round(float(insider) * 100, 1) if insider else None

        mc = info.get("marketCap")
        if mc:
            result["market_cap"] = _fmt_inr(mc, is_indian)

        eps_fwd = info.get("forwardEps")
        eps_tr  = info.get("trailingEps")
        if eps_fwd and eps_tr:
            trend = "↑ Growing" if eps_fwd > eps_tr else "↓ Declining"
            result["earnings"] = f"{trend} ({eps_tr:.2f}→{eps_fwd:.2f})"
        elif eps_tr:
            result["earnings"] = f"EPS {eps_tr:.2f}"
        else:
            # Fallback: compare last two years of annual net income
            try:
                af = stock.financials  # annual, columns = year-end dates newest first
                if af is not None and not af.empty and "Net Income" in af.index:
                    ni = af.loc["Net Income"].dropna()
                    if len(ni) >= 2:
                        latest_ni = float(ni.iloc[0])
                        prev_ni   = float(ni.iloc[1])
                        trend = "↑ Growing" if latest_ni > prev_ni else "↓ Declining"
                        ni_str = _fmt_inr(abs(latest_ni), is_indian)
                        result["earnings"] = f"{trend} (Net Income: {ni_str})"
            except Exception:
                pass

        # Industry / sector P/E for health check comparison
        result["industry_pe"] = info.get("industryPe") or info.get("sectorPe") or None

        # ── Quarterly financials ───────────────────────────────
        try:
            qf = stock.quarterly_financials
            if qf is not None and not qf.empty:
                # Sort columns newest-first so quarters appear in correct order
                qf = qf.sort_index(axis=1, ascending=False)
                cols = list(qf.columns[:4])  # latest 4 quarters
                col_labels = [
                    c.strftime("%b '%y") if hasattr(c, "strftime") else str(c)
                    for c in cols
                ]

                def _fmt_cell(val):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return None
                    if is_indian:
                        cr = val / 1e7
                        if abs(cr) >= 1e5:
                            return val / 1e7, f"₹{cr/1e5:.2f}LCr"
                        return val / 1e7, f"₹{cr:,.0f}Cr"
                    else:
                        if abs(val) >= 1e9:
                            return val / 1e9, f"${val/1e9:.2f}B"
                        if abs(val) >= 1e6:
                            return val / 1e6, f"${val/1e6:.1f}M"
                        return val, f"{val:,.0f}"

                rows_data = {}

                for label, key in [
                    ("Revenue", "Total Revenue"),
                    ("Net Profit", "Net Income"),
                    ("Operating Income", "Operating Income"),
                    ("Interest Expense", "Interest Expense"),
                ]:
                    if key in qf.index:
                        vals = []
                        for c in cols:
                            v = qf.loc[key, c]
                            parsed = _fmt_cell(v) if not pd.isna(v) else None
                            vals.append(parsed)
                        rows_data[label] = vals

                # Revenue growth: latest vs oldest of the 4 quarters
                if "Revenue" in rows_data:
                    rev_vals = [v[0] if v else None for v in rows_data["Revenue"]]
                    latest = rev_vals[0] if rev_vals else None
                    oldest = next((v for v in reversed(rev_vals) if v is not None), None)
                    if latest and oldest and oldest != 0:
                        result["revenue_growth"] = round((latest - oldest) / abs(oldest) * 100, 1)

                # Bar chart data for Net Profit (last 4 quarters, oldest→newest)
                if "Net Profit" in rows_data:
                    bar_data = []
                    for lbl, v in zip(reversed(col_labels), reversed(rows_data["Net Profit"])):
                        bar_data.append({
                            "Quarter": lbl,
                            "Net Profit (Cr)" if is_indian else "Net Profit (B)": round(v[0], 2) if v else 0,
                        })
                    result["quarterly_bar_data"] = bar_data

                # Derived rows
                op_margin_row = []
                if "Revenue" in rows_data and "Operating Income" in rows_data:
                    for rev, oi in zip(rows_data["Revenue"], rows_data["Operating Income"]):
                        if rev and oi and rev[0] != 0:
                            pct = oi[0] / rev[0] * 100
                            op_margin_row.append((pct, f"{pct:.1f}%"))
                        else:
                            op_margin_row.append(None)

                icr_row = []
                if "Operating Income" in rows_data and "Interest Expense" in rows_data:
                    for oi, ie in zip(rows_data["Operating Income"], rows_data["Interest Expense"]):
                        if oi and ie and ie[0] != 0:
                            icr = abs(oi[0] / ie[0])
                            icr_row.append((icr, f"{icr:.1f}x"))
                        else:
                            icr_row.append(None)

                # Build styled HTML table
                def _cell_style(label, raw_val, display_str, all_raws):
                    bg = "transparent"
                    bold = False
                    valid = [r for r in all_raws if r is not None]
                    if valid:
                        if raw_val == max(valid):
                            bold = True
                        if raw_val == min(valid):
                            bold = True
                    if label == "Net Profit" and raw_val is not None:
                        bg = "rgba(34,197,94,0.15)" if raw_val > 0 else "rgba(239,68,68,0.15)"
                    txt = f"<b>{display_str}</b>" if bold else display_str
                    return txt, bg

                th_style = ("background:#1e293b; color:#94a3b8; font-size:0.72rem; "
                            "font-weight:700; letter-spacing:0.8px; padding:6px 10px; "
                            "text-align:center; border-bottom:1px solid #334155;")
                td_style_base = ("font-size:0.82rem; padding:6px 10px; text-align:center; "
                                 "border-bottom:1px solid rgba(255,255,255,0.04);")

                header = "".join(f"<th style='{th_style}'>{lbl}</th>" for lbl in col_labels)
                html = (
                    "<table style='width:100%; border-collapse:collapse; "
                    "background:rgba(15,23,42,0.6); border-radius:10px; overflow:hidden;'>"
                    f"<thead><tr><th style='{th_style} text-align:left;'>Metric</th>{header}</tr></thead>"
                    "<tbody>"
                )

                display_rows = [
                    ("Revenue",          rows_data.get("Revenue", [None]*4),          "#e2e8f0"),
                    ("Net Profit",       rows_data.get("Net Profit", [None]*4),        "#e2e8f0"),
                    ("Operating Margin", op_margin_row if op_margin_row else [None]*4, "#38bdf8"),
                    ("Interest Coverage",icr_row if icr_row else [None]*4,             "#a78bfa"),
                ]

                for row_label, vals, label_color in display_rows:
                    raw_floats = [v[0] if v else None for v in vals]
                    html += f"<tr><td style='{td_style_base} text-align:left; color:{label_color}; font-weight:600;'>{row_label}</td>"
                    for v in vals:
                        if v is None:
                            html += f"<td style='{td_style_base} color:#475569;'>—</td>"
                        else:
                            raw_f, disp = v
                            cell_txt, bg = _cell_style(row_label, raw_f, disp, raw_floats)
                            html += f"<td style='{td_style_base} background:{bg}; color:#e2e8f0;'>{cell_txt}</td>"
                    html += "</tr>"

                html += "</tbody></table>"
                result["quarterly_html"] = html

        except Exception:
            pass

    except Exception:
        pass
    return result


def _gauge_bar(value, low_thresh, high_thresh, low_label, mid_label, high_label,
               fmt="{:.1f}", invert=False) -> str:
    """
    Render a small 3-zone horizontal gauge bar.
    invert=True means low value = bad (e.g. negative D/E is unusual).
    """
    if value is None:
        return "<span style='color:#475569; font-size:0.8rem;'>N/A</span>"

    display = fmt.format(value)

    # Clamp position 0–100%
    span = high_thresh - low_thresh
    if span <= 0:
        pos = 50
    else:
        pos = max(0, min(100, (value - low_thresh) / span * 100))

    if value <= low_thresh:
        zone_color = "#22c55e" if not invert else "#ef4444"
        zone_label = low_label
    elif value <= high_thresh:
        zone_color = "#f59e0b"
        zone_label = mid_label
    else:
        zone_color = "#ef4444" if not invert else "#22c55e"
        zone_label = high_label

    return f"""
    <div style="margin-top:0.15rem;">
      <span style="font-size:1.3rem; font-weight:800; color:{zone_color};">{display}</span>
      <span style="font-size:0.72rem; color:{zone_color}; margin-left:6px;">{zone_label}</span>
      <div style="background:linear-gradient(90deg,#22c55e 0%,#22c55e 33%,
                  #f59e0b 33%,#f59e0b 66%,#ef4444 66%,#ef4444 100%);
                  border-radius:999px; height:6px; margin-top:4px; position:relative;">
        <div style="position:absolute; left:{pos:.0f}%; top:-3px;
                    width:3px; height:12px; background:#fff;
                    border-radius:2px; transform:translateX(-50%);"></div>
      </div>
    </div>"""


def display_fundamental_data(recommendation: Recommendation):
    if not recommendation.fundamental_summary:
        st.warning("Fundamental analysis data not available")
        return

    summary = recommendation.fundamental_summary
    metrics = parse_pipe_summary(summary)

    # Company name — first pipe segment with no colon
    company_name = "N/A"
    parts = summary.split("|")
    if parts:
        first = parts[0].strip()
        if ":" not in first:
            company_name = first

    extra = fetch_extra_fundamentals(recommendation.ticker)
    is_indian = extra.get("is_indian", False)
    cur = "₹" if is_indian else "$"

    with st.expander("💼 Fundamental Analysis", expanded=True):
        # ── Header ────────────────────────────────────────────
        if company_name != "N/A":
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);"
                f"border-radius:10px; padding:0.5rem 1rem; margin-bottom:0.8rem; display:inline-block;'>"
                f"<span style='font-size:1rem; font-weight:700; color:#e2e8f0;'>🏢 {company_name}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ── Row 1: Market Cap + P/E + P/B ─────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            mc = extra.get("market_cap") or "N/A"
            st.markdown(
                f"<div class='indicator-card'>"
                f"<div class='indicator-title'>Market Cap</div>"
                f"<div style='font-size:1.3rem; font-weight:800; color:#e2e8f0;'>{mc}</div>"
                f"</div>", unsafe_allow_html=True
            )
        with c2:
            pe_val = extra.get("pe")
            pe_gauge = _gauge_bar(
                pe_val, low_thresh=15, high_thresh=30,
                low_label="Undervalued", mid_label="Fair", high_label="Overvalued",
                fmt="{:.1f}x"
            )
            st.markdown(
                f"<div class='indicator-card'>"
                f"<div class='indicator-title'>P/E Ratio</div>"
                f"{pe_gauge}"
                f"</div>", unsafe_allow_html=True
            )
        with c3:
            de_val = extra.get("de")
            de_gauge = _gauge_bar(
                de_val, low_thresh=0.5, high_thresh=1.5,
                low_label="Low Debt", mid_label="Moderate", high_label="High Debt",
                fmt="{:.2f}x"
            )
            st.markdown(
                f"<div class='indicator-card'>"
                f"<div class='indicator-title'>Debt / Equity</div>"
                f"{de_gauge}"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Row 2: ROE + Dividend Yield + Promoter Holding ────
        c4, c5, c6 = st.columns(3)
        with c4:
            roe = extra.get("roe")
            roe_color = "#22c55e" if roe and roe > 15 else ("#f59e0b" if roe and roe > 8 else "#ef4444")
            roe_label = "Strong" if roe and roe > 15 else ("Moderate" if roe and roe > 8 else "Weak")
            roe_disp  = f"{roe:.1f}%" if roe is not None else "N/A"
            st.markdown(
                f"<div class='indicator-card'>"
                f"<div class='indicator-title'>ROE (Return on Equity)</div>"
                f"<div style='font-size:1.3rem; font-weight:800; color:{roe_color};'>{roe_disp}</div>"
                f"<div class='indicator-sub'>{roe_label}</div>"
                f"</div>", unsafe_allow_html=True
            )
        with c5:
            dy = extra.get("div_yield")
            dy_color = "#22c55e" if dy and dy > 2 else ("#f59e0b" if dy and dy > 0.5 else "#94a3b8")
            dy_disp  = f"{dy:.2f}%" if dy is not None else "N/A"
            dy_label = "High Yield" if dy and dy > 2 else ("Moderate" if dy and dy > 0.5 else "Low / None")
            st.markdown(
                f"<div class='indicator-card'>"
                f"<div class='indicator-title'>Dividend Yield</div>"
                f"<div style='font-size:1.3rem; font-weight:800; color:{dy_color};'>{dy_disp}</div>"
                f"<div class='indicator-sub'>{dy_label}</div>"
                f"</div>", unsafe_allow_html=True
            )
        with c6:
            ph = extra.get("promoter_holding")
            ph_color = "#22c55e" if ph and ph > 50 else ("#f59e0b" if ph and ph > 25 else "#ef4444")
            ph_label = "High Conviction" if ph and ph > 50 else ("Moderate" if ph and ph > 25 else "Low")
            ph_disp  = f"{ph:.1f}%" if ph is not None else "N/A"
            st.markdown(
                f"<div class='indicator-card'>"
                f"<div class='indicator-title'>Promoter / Insider Holding</div>"
                f"<div style='font-size:1.3rem; font-weight:800; color:{ph_color};'>{ph_disp}</div>"
                f"<div class='indicator-sub'>{ph_label}</div>"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # ── Earnings Bar Chart (Net Profit last 4 quarters) ───
        bar_data = extra.get("quarterly_bar_data")
        profit_col = "Net Profit (Cr)" if is_indian else "Net Profit (B)"
        if bar_data:
            st.markdown(
                "<p style='font-size:0.82rem; font-weight:700; color:#94a3b8; "
                "letter-spacing:0.8px; text-transform:uppercase; margin-bottom:0.2rem;'>"
                "📊 Net Profit — Last 4 Quarters</p>",
                unsafe_allow_html=True
            )
            bar_df = pd.DataFrame(bar_data).set_index("Quarter")
            # Color bars green/red based on sign
            st.bar_chart(bar_df[[profit_col]], height=200, use_container_width=True)
        else:
            earn = extra.get("earnings") or "N/A"
            earn_color = "#22c55e" if "↑" in earn else ("#ef4444" if "↓" in earn else "#e2e8f0")
            st.markdown(
                f"<div class='indicator-card'>"
                f"<div class='indicator-title'>Earnings Trend</div>"
                f"<div style='font-size:1.3rem; font-weight:800; color:{earn_color};'>{earn}</div>"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Financial Health Score ─────────────────────────────
        st.markdown(
            "<p style='font-size:0.82rem; font-weight:700; color:#94a3b8; "
            "letter-spacing:0.8px; text-transform:uppercase; margin-bottom:0.4rem;'>"
            "🏥 Financial Health Check</p>",
            unsafe_allow_html=True
        )
        de_val      = extra.get("de")
        pe_val      = extra.get("pe")
        ind_pe      = extra.get("industry_pe")
        rev_growth  = extra.get("revenue_growth")

        checks = []
        # 1. Debt/Equity
        if de_val is not None:
            ok = de_val < 1.0
            checks.append(("✅" if ok else "❌",
                            f"Debt/Equity = {de_val:.2f}x",
                            "Below 1.0 — healthy leverage" if ok else "Above 1.0 — elevated debt",
                            ok))
        else:
            checks.append(("⚪", "Debt/Equity", "Data unavailable", None))

        # 2. P/E vs Industry
        if pe_val is not None and ind_pe:
            ok = pe_val < float(ind_pe)
            checks.append(("✅" if ok else "❌",
                            f"P/E {pe_val:.1f}x vs Industry {float(ind_pe):.1f}x",
                            "Trading below sector average" if ok else "Premium to sector average",
                            ok))
        elif pe_val is not None:
            ok = pe_val < 25
            checks.append(("✅" if ok else "⚠️",
                            f"P/E = {pe_val:.1f}x",
                            "Reasonable valuation" if ok else "Elevated valuation",
                            ok))
        else:
            checks.append(("⚪", "P/E Ratio", "Data unavailable", None))

        # 3. Revenue Growth
        if rev_growth is not None:
            ok = rev_growth >= 0
            checks.append(("✅" if ok else "❌",
                            f"Revenue Growth = {rev_growth:+.1f}%",
                            "Positive growth trend" if ok else "Revenue declining — caution",
                            ok))
        else:
            checks.append(("⚪", "Revenue Growth", "Data unavailable", None))

        health_html = "<div style='display:flex; flex-direction:column; gap:6px;'>"
        for icon, label, desc, ok in checks:
            icon_color = "#22c55e" if ok is True else ("#ef4444" if ok is False else "#64748b")
            health_html += (
                f"<div style='background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.06); "
                f"border-radius:8px; padding:0.45rem 0.9rem; display:flex; align-items:center; gap:10px;'>"
                f"<span style='font-size:1.1rem;'>{icon}</span>"
                f"<div>"
                f"<span style='font-size:0.83rem; font-weight:700; color:{icon_color};'>{label}</span>"
                f"<span style='font-size:0.75rem; color:#64748b; margin-left:8px;'>{desc}</span>"
                f"</div></div>"
            )
        health_html += "</div>"
        st.markdown(health_html, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # ── Quarterly Table ────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.82rem; font-weight:700; color:#94a3b8; "
            "letter-spacing:0.8px; text-transform:uppercase; margin-bottom:0.4rem;'>"
            "📋 Quarterly Financials (Last 4 Quarters)</p>",
            unsafe_allow_html=True
        )
        if extra["quarterly_html"]:
            st.markdown(extra["quarterly_html"], unsafe_allow_html=True)
        else:
            st.caption("Quarterly data unavailable")


def fetch_news_headlines(ticker: str, max_items: int = 5) -> list:
    """
    Fetch news from multiple sources:
    1. yfinance built-in news
    2. Google News RSS
    3. Yahoo Finance RSS
    Each item: {title, link, publisher, published_ts, sentiment}
    """
    import time as _time

    POSITIVE_WORDS = {
        "surge", "soar", "rally", "gain", "profit", "beat", "record", "high",
        "growth", "strong", "upgrade", "buy", "bullish", "rise", "jump",
        "outperform", "positive", "boost", "win", "success", "expand",
    }
    NEGATIVE_WORDS = {
        "fall", "drop", "crash", "loss", "miss", "low", "weak", "sell",
        "bearish", "decline", "cut", "downgrade", "risk", "concern", "warn",
        "negative", "plunge", "slump", "fear", "debt", "default", "fraud",
    }

    def _score_title(title: str) -> str:
        t = title.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in t)
        neg = sum(1 for w in NEGATIVE_WORDS if w in t)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    def _time_ago(ts) -> str:
        """Convert unix timestamp or struct_time to '2h ago' string."""
        try:
            import calendar
            if hasattr(ts, "tm_year"):          # struct_time from feedparser
                epoch = calendar.timegm(ts)
            else:
                epoch = int(ts)
            diff = int(_time.time()) - epoch
            if diff < 3600:
                return f"{diff // 60}m ago"
            if diff < 86400:
                return f"{diff // 3600}h ago"
            return f"{diff // 86400}d ago"
        except Exception:
            return ""

    headlines = []
    seen_titles: set = set()

    # ── Source 1: yfinance ────────────────────────────────────
    try:
        import yfinance as yf
        raw_news = yf.Ticker(ticker).news or []
        for item in raw_news[:10]:
            content   = item.get("content", item)
            title     = content.get("title", "")
            link      = (content.get("canonicalUrl") or {}).get("url", "") or content.get("link", "")
            publisher = (content.get("provider") or {}).get("displayName", "") or content.get("publisher", "")
            pub_ts    = content.get("pubDate") or content.get("providerPublishTime") or 0
            if title and title not in seen_titles:
                seen_titles.add(title)
                headlines.append({
                    "title": title, "link": link,
                    "publisher": publisher or "Yahoo Finance",
                    "time_ago": _time_ago(pub_ts),
                    "sentiment": _score_title(title),
                })
    except Exception:
        pass

    # ── Source 2: Google News RSS ─────────────────────────────
    try:
        import feedparser
        # Use ticker base name for better results
        query = ticker.replace(".NS", "").replace(".BO", "")
        gn_url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(gn_url)
        for entry in feed.entries[:8]:
            title = entry.get("title", "")
            # Google News titles often have " - Source" suffix
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                clean_title = parts[0].strip()
                source = parts[1].strip()
            else:
                clean_title = title
                source = entry.get("source", {}).get("title", "Google News")
            link = entry.get("link", "")
            pub  = entry.get("published_parsed")
            if clean_title and clean_title not in seen_titles:
                seen_titles.add(clean_title)
                headlines.append({
                    "title": clean_title, "link": link,
                    "publisher": source,
                    "time_ago": _time_ago(pub) if pub else "",
                    "sentiment": _score_title(clean_title),
                })
    except Exception:
        pass

    # ── Source 3: Yahoo Finance RSS ───────────────────────────
    try:
        import feedparser
        yf_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed2  = feedparser.parse(yf_rss)
        for entry in feed2.entries[:6]:
            title = entry.get("title", "")
            link  = entry.get("link", "")
            pub   = entry.get("published_parsed")
            if title and title not in seen_titles:
                seen_titles.add(title)
                headlines.append({
                    "title": title, "link": link,
                    "publisher": "Yahoo Finance",
                    "time_ago": _time_ago(pub) if pub else "",
                    "sentiment": _score_title(title),
                })
    except Exception:
        pass

    return headlines[:max_items]


def _compute_composite_sentiment(base_score: float, ticker: str) -> dict:
    """
    Blend news score (60%) + price momentum (20%) + volume spike (20%).
    Returns dict with: score (0-1), label, color, icon, description
    """
    try:
        import yfinance as yf
        import numpy as np

        df = yf.Ticker(ticker).history(period="3mo")
        if df.empty or len(df) < 20:
            raise ValueError("insufficient data")

        close  = df["Close"]
        volume = df["Volume"]

        # Price momentum score (0–1)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi   = float(100 - (100 / (1 + rs.iloc[-1])))
        ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())
        price_now = float(close.iloc[-1])

        # RSI: oversold (<30) → bullish boost, overbought (>70) → bearish drag
        rsi_score = 0.5
        if rsi < 30:
            rsi_score = 0.75
        elif rsi > 70:
            rsi_score = 0.25
        else:
            rsi_score = 0.5 + (rsi - 50) / 100  # linear between 0.3–0.7

        # Price vs MA50
        ma_score = 0.65 if price_now > ma50 else 0.35

        momentum_score = (rsi_score + ma_score) / 2

        # Volume spike score (0–1)
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_now = float(volume.iloc[-1])
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
        # High volume amplifies the direction of price momentum
        if vol_ratio > 1.5:
            vol_score = 0.7 if momentum_score > 0.5 else 0.3
        elif vol_ratio < 0.6:
            vol_score = 0.5  # low volume = indecisive
        else:
            vol_score = 0.5

        composite = (base_score * 0.60) + (momentum_score * 0.20) + (vol_score * 0.20)
        composite = max(0.0, min(1.0, composite))

    except Exception:
        composite = base_score
        vol_ratio = 1.0
        rsi = 50.0

    # Map composite → dynamic label
    if composite >= 0.80:
        label, icon, color, desc = "Strong Bullish",    "🚀", "#22c55e", "Strong buying momentum"
    elif composite >= 0.65:
        label, icon, color, desc = "Cautious Bullish",  "📈", "#4ade80", "Mild positive trend"
    elif composite >= 0.55:
        label, icon, color, desc = "Volatile Neutral",  "⚡", "#f59e0b", "Mixed signals, watch closely"
    elif composite >= 0.40:
        label, icon, color, desc = "Cautious Bearish",  "📉", "#fb923c", "Mild selling pressure"
    elif composite >= 0.25:
        label, icon, color, desc = "Panic Selling",     "⚠️", "#ef4444", "Heavy selling detected"
    else:
        label, icon, color, desc = "Extreme Bearish",   "💀", "#dc2626", "Extreme negative sentiment"

    return {
        "score": composite, "label": label, "icon": icon,
        "color": color, "desc": desc,
    }


def display_sentiment_data(recommendation: Recommendation):
    if not recommendation.sentiment_summary:
        st.warning("Sentiment analysis data not available")
        return

    summary = recommendation.sentiment_summary

    # Extract base score from agent summary
    base_score = 0.5
    score_match = re.search(r"Score:\s*([\d.]+)", summary)
    if score_match:
        try:
            base_score = float(score_match.group(1))
        except ValueError:
            pass

    # Composite sentiment (news + momentum + volume)
    sent  = _compute_composite_sentiment(base_score, recommendation.ticker)
    score = sent["score"]
    label = sent["label"]
    icon  = sent["icon"]
    color = sent["color"]
    desc  = sent["desc"]

    st.markdown(
        "<h4 style='margin-bottom:0.6rem; color:#e2e8f0;'>💭 Sentiment Analysis</h4>",
        unsafe_allow_html=True
    )

    # ── Fetch headlines first (needed for Claude analysis) ────
    headlines = fetch_news_headlines(recommendation.ticker, max_items=15)

    # ── Claude AI headline analysis ───────────────────────────
    ai_analysis = {}
    if headlines:
        import json as _json
        try:
            hl_json = _json.dumps([{"title": h["title"]} for h in headlines])
            ai_analysis = analyze_headlines_with_claude(recommendation.ticker, hl_json)
        except Exception:
            ai_analysis = {}

    ai_avg_score = ai_analysis.get("avg_score", 0.0)   # -1 to 1
    ai_insight   = ai_analysis.get("insight", "")
    ai_scores    = ai_analysis.get("scores", {})

    # ── Detect technical vs news conflict ────────────────────
    # Technical direction from recommendation signal
    tech_signal = recommendation.signal  # BUY / SELL / HOLD
    news_is_bearish = ai_avg_score < -0.15
    news_is_bullish = ai_avg_score > 0.15
    conflict = (
        (tech_signal == "BUY"  and news_is_bearish) or
        (tech_signal == "SELL" and news_is_bullish)
    )

    # Adjusted confidence: penalise by up to 15 pts when conflicting
    base_conf = recommendation.confidence_score or 50
    if conflict:
        penalty = min(15, int(abs(ai_avg_score) * 20))
        adj_conf = max(0, base_conf - penalty)
    else:
        adj_conf = base_conf

    # ── Conflict warning banner ───────────────────────────────
    if conflict:
        warn_dir = "Bearish" if news_is_bearish else "Bullish"
        st.markdown(
            f"<div style='background:rgba(245,158,11,0.12); border:1px solid #f59e0b55; "
            f"border-left:4px solid #f59e0b; border-radius:8px; padding:0.6rem 1rem; "
            f"margin-bottom:0.8rem;'>"
            f"<span style='font-size:0.85rem; font-weight:700; color:#f59e0b;'>⚠️ Sentiment Conflict Detected</span><br>"
            f"<span style='font-size:0.78rem; color:#94a3b8;'>"
            f"Technical signal is <b style='color:#e2e8f0;'>{tech_signal}</b> but news sentiment is "
            f"<b style='color:#f59e0b;'>{warn_dir}</b> (AI score: {ai_avg_score:+.2f}). "
            f"Confidence adjusted from <b>{base_conf}</b> → <b style='color:#f59e0b;'>{adj_conf}</b>."
            f"</span></div>",
            unsafe_allow_html=True
        )

    # ── AI Insight banner ─────────────────────────────────────
    if ai_insight:
        insight_color = "#22c55e" if ai_avg_score > 0.1 else ("#ef4444" if ai_avg_score < -0.1 else "#f59e0b")
        insight_icon  = "🟢" if ai_avg_score > 0.1 else ("🔴" if ai_avg_score < -0.1 else "🟡")
        st.markdown(
            f"<div style='background:rgba(99,102,241,0.10); border:1px solid #6366f155; "
            f"border-radius:8px; padding:0.55rem 1rem; margin-bottom:0.8rem;'>"
            f"<span style='font-size:0.72rem; font-weight:700; color:#94a3b8; "
            f"letter-spacing:0.8px; text-transform:uppercase;'>🤖 AI News Insight</span><br>"
            f"<span style='font-size:0.85rem; color:#e2e8f0;'>{insight_icon} {ai_insight}</span>"
            f"<span style='font-size:0.72rem; color:#475569; margin-left:8px;'>"
            f"(avg score: <b style='color:{insight_color};'>{ai_avg_score:+.2f}</b>)</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Sentiment Meter (speedometer-style SVG) ───────────────
    needle_deg = -90 + int(score * 180)
    meter_html = f"""
    <div style="text-align:center; margin-bottom:0.6rem;">
      <svg viewBox="0 0 200 110" width="220" style="overflow:visible;">
        <path d="M 20 100 A 80 80 0 0 1 60 27" stroke="#ef4444" stroke-width="14"
              fill="none" stroke-linecap="round"/>
        <path d="M 60 27 A 80 80 0 0 1 100 20" stroke="#f97316" stroke-width="14"
              fill="none" stroke-linecap="round"/>
        <path d="M 100 20 A 80 80 0 0 1 140 27" stroke="#f59e0b" stroke-width="14"
              fill="none" stroke-linecap="round"/>
        <path d="M 140 27 A 80 80 0 0 1 170 60" stroke="#84cc16" stroke-width="14"
              fill="none" stroke-linecap="round"/>
        <path d="M 170 60 A 80 80 0 0 1 180 100" stroke="#22c55e" stroke-width="14"
              fill="none" stroke-linecap="round"/>
        <g transform="translate(100,100) rotate({needle_deg})">
          <line x1="0" y1="0" x2="0" y2="-68"
                stroke="{color}" stroke-width="3" stroke-linecap="round"/>
          <circle cx="0" cy="0" r="6" fill="{color}"/>
        </g>
        <text x="100" y="108" text-anchor="middle"
              font-size="13" font-weight="800" fill="{color}"
              font-family="Inter,Segoe UI,sans-serif">{label}</text>
      </svg>
      <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.1rem;">{desc}</div>
      <div style="font-size:0.72rem; color:#475569; margin-top:0.2rem;">
        Composite Score: <b style="color:{color};">{score:.2f}</b> / 1.00
        &nbsp;|&nbsp; News 60% · Momentum 20% · Volume 20%
        {"&nbsp;|&nbsp; <b style='color:#f59e0b;'>Adj. Confidence: " + str(adj_conf) + "</b>" if conflict else ""}
      </div>
    </div>"""
    st.markdown(meter_html, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── News Cards with AI scores ─────────────────────────────
    st.markdown(
        "<p style='font-weight:700; font-size:0.88rem; color:#94a3b8; "
        "letter-spacing:0.8px; text-transform:uppercase; margin-bottom:0.5rem;'>"
        "📰 Latest News</p>",
        unsafe_allow_html=True
    )

    if headlines:
        IMPACT_COLORS = {
            "positive": ("#22c55e", "rgba(34,197,94,0.12)", "✅ Positive"),
            "negative": ("#ef4444", "rgba(239,68,68,0.12)", "🔴 Negative"),
            "neutral":  ("#64748b", "rgba(100,116,139,0.10)", "⚪ Neutral"),
        }
        cards_html = "<div style='max-height:420px; overflow-y:auto; padding-right:4px;'>"
        for item in headlines:
            s = item.get("sentiment", "neutral")
            imp_color, imp_bg, imp_label = IMPACT_COLORS.get(s, IMPACT_COLORS["neutral"])
            pub   = item.get("publisher", "")
            t_ago = item.get("time_ago", "")
            title = item["title"]
            link  = item.get("link", "")

            # AI score badge for this headline
            ai_score = ai_scores.get(title)
            if ai_score is not None:
                sc_color = "#22c55e" if ai_score > 0.1 else ("#ef4444" if ai_score < -0.1 else "#94a3b8")
                ai_badge = (
                    f"<span style='background:rgba(99,102,241,0.15); border:1px solid #6366f155; "
                    f"border-radius:4px; padding:1px 6px; font-size:0.68rem; "
                    f"color:{sc_color}; font-weight:700;'>AI {ai_score:+.2f}</span>"
                )
            else:
                ai_badge = ""

            title_html = (
                f"<a href='{link}' target='_blank' "
                f"style='color:#e2e8f0; text-decoration:none; font-size:0.85rem; "
                f"font-weight:600; line-height:1.4;'>{title}</a>"
                if link
                else f"<span style='color:#e2e8f0; font-size:0.85rem; font-weight:600;'>{title}</span>"
            )
            pub_tag = (
                f"<span style='background:#1e293b; border:1px solid #334155; "
                f"border-radius:4px; padding:1px 6px; font-size:0.68rem; color:#94a3b8;'>"
                f"{pub}</span>" if pub else ""
            )
            time_tag = (
                f"<span style='color:#475569; font-size:0.68rem;'>{t_ago}</span>"
                if t_ago else ""
            )
            impact_tag = (
                f"<span style='background:{imp_bg}; border:1px solid {imp_color}44; "
                f"border-radius:4px; padding:1px 7px; font-size:0.68rem; color:{imp_color}; "
                f"font-weight:600;'>{imp_label}</span>"
            )
            cards_html += (
                f"<div style='background:#0f172a; border:1px solid #1e293b; "
                f"border-left:3px solid {imp_color}; border-radius:8px; "
                f"padding:0.55rem 0.8rem; margin-bottom:0.4rem;'>"
                f"{title_html}"
                f"<div style='margin-top:0.35rem; display:flex; gap:6px; flex-wrap:wrap; "
                f"align-items:center;'>{pub_tag}{time_tag}{impact_tag}{ai_badge}</div>"
                f"</div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        # Social Buzz score
        pos_count  = sum(1 for h in headlines if h.get("sentiment") == "positive")
        neg_count  = sum(1 for h in headlines if h.get("sentiment") == "negative")
        total      = len(headlines)
        buzz_score = round((pos_count - neg_count) / total * 100) if total else 0
        buzz_color = "#22c55e" if buzz_score > 10 else ("#ef4444" if buzz_score < -10 else "#f59e0b")
        buzz_label = "Bullish Buzz" if buzz_score > 10 else ("Bearish Buzz" if buzz_score < -10 else "Mixed Buzz")
        st.markdown(
            f"<div style='margin-top:0.6rem; background:rgba(255,255,255,0.03); "
            f"border:1px solid rgba(255,255,255,0.07); border-radius:8px; "
            f"padding:0.5rem 0.9rem; display:flex; justify-content:space-between; align-items:center;'>"
            f"<span style='font-size:0.78rem; color:#94a3b8;'>📡 Social Buzz Score</span>"
            f"<span style='font-size:1rem; font-weight:800; color:{buzz_color};'>"
            f"{buzz_score:+d}% &nbsp; {buzz_label}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.caption("No news available")

    st.markdown("---")

    # ── Store conflict info in session for AI Reasoning section ──
    if conflict:
        st.session_state["news_conflict"] = {
            "signal":    tech_signal,
            "direction": "Bearish" if news_is_bearish else "Bullish",
            "ai_score":  ai_avg_score,
            "adj_conf":  adj_conf,
            "insight":   ai_insight,
        }
    else:
        st.session_state.pop("news_conflict", None)


def format_ist_timestamp(utc_timestamp: str) -> str:
    """Convert ISO 8601 UTC timestamp to IST and format as '17 Mar 2026, 03:35 PM'."""
    try:
        from datetime import timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        # Handle both 'Z' suffix and '+00:00' suffix
        ts = utc_timestamp.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(ts)
        dt_ist = dt_utc.astimezone(IST)
        return dt_ist.strftime("%d %b %Y, %I:%M %p") + " IST"
    except Exception:
        return utc_timestamp  # fallback to raw string if parsing fails


def _sanitise_pdf(txt) -> str:
    """Strip/replace characters that crash Helvetica in fpdf2. Safely handles None."""
    import re as _re
    if txt is None:
        return ""
    txt = str(txt)
    # Map Indian Rupee symbol to ASCII equivalent before latin-1 encoding
    txt = txt.replace('\u20b9', 'Rs.')  # ₹ → Rs.
    # Strip non-latin-1 (Devanagari, emoji, etc.) — keep only printable ASCII + latin-1
    txt = txt.encode('latin-1', errors='replace').decode('latin-1')
    # Remove replacement char artifacts
    txt = _re.sub(r'[\x80-\x9f]', '', txt)
    return txt.strip()
def _safe(val, fmt=None, fallback="N/A"):
    """Return formatted value or fallback if None/NaN."""
    if val is None:
        return fallback
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return fallback
        return fmt.format(val) if fmt else str(val)
    except Exception:
        return fallback


def fetch_peer_comparison(ticker: str) -> list:
    """
    Fetch top 3 sector peers and compare P/E, Debt/Equity, Market Cap, 1-Year Return.
    Currency: INR (₹ Cr) for Indian tickers, USD for others.
    Returns list of dicts: [{name, ticker, pe, de, market_cap, return_1y, is_current}]
    """
    try:
        import yfinance as yf

        is_indian = ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")

        stock = yf.Ticker(ticker)
        info  = stock.info or {}
        sector   = info.get("sector", "")
        industry = info.get("industry", "")

        PEER_MAP = {
            # ── Indian Infrastructure / PSU ───────────────────
            "Railways":                ["RVNL.NS", "IRCON.NS", "RITES.NS", "RAILTEL.NS", "IRFC.NS"],
            "Defence":                 ["HAL.NS", "BEL.NS", "MAZDOCK.NS", "BDL.NS", "BEML.NS"],
            "Engineering & Construction": ["LT.NS", "NBCC.NS", "RVNL.NS", "IRCON.NS", "KEC.NS", "KALPATPOWR.NS"],
            "Power Finance":           ["PFC.NS", "RECLTD.NS", "IREDA.NS", "POWERGRID.NS"],
            # ── Energy ───────────────────────────────────────
            "Oil & Gas Integrated":    ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "HINDPETRO.NS"],
            "Oil & Gas E&P":           ["ONGC.NS", "OIL.NS", "GAIL.NS", "GSPL.NS"],
            "Utilities—Regulated":     ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "CESC.NS", "TORNTPOWER.NS"],
            "Fertilizers":             ["GSFC.NS", "GNFC.NS", "CHAMBLFERT.NS", "COROMANDEL.NS", "DEEPAKFERT.NS"],
            # ── Indian IT ────────────────────────────────────
            "Information Technology":  ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
            "IT Services":             ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
            # ── US Software ──────────────────────────────────
            "Software—Application":    ["MSFT", "ORCL", "SAP", "CRM", "ADBE", "NOW"],
            "Software—Infrastructure": ["MSFT", "ORCL", "IBM", "CSCO", "VMW", "HPE"],
            "Semiconductors":          ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN"],
            "Consumer Electronics":    ["AAPL", "SONY", "SSNLF", "DELL", "HPQ"],
            # ── US Auto / EV ─────────────────────────────────
            "Auto Manufacturers—US":   ["TSLA", "F", "GM", "RIVN", "LCID", "STLA"],
            "Electric Vehicles":       ["TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI"],
            # ── US Banks ─────────────────────────────────────
            "Banks—US":                ["JPM", "BAC", "GS", "MS", "C", "WFC"],
            "Banks—Regional":          ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "INDUSINDBK.NS"],
            "Banks—Diversified":       ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "INDUSINDBK.NS"],
            "Banks—Global":            ["JPM", "BAC", "GS", "MS", "C", "WFC"],
            "Insurance—Life":          ["HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "LICI.NS", "MAXLIFE.NS"],
            "Asset Management":        ["HDFCAMC.NS", "NIPPONLIFE.NS", "UTIAMC.NS", "360ONE.NS"],
            "NBFC":                    ["BAJFINANCE.NS", "BAJAJFINSV.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS", "MANAPPURAM.NS"],
            # ── Auto ─────────────────────────────────────────
            "Auto Manufacturers":      ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
            "Auto Components":         ["BOSCHLTD.NS", "MOTHERSON.NS", "BHARATFORG.NS", "APOLLOTYRE.NS", "BALKRISIND.NS"],
            # ── Pharma ───────────────────────────────────────
            "Drug Manufacturers":      ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "LUPIN.NS"],
            "Pharmaceuticals":         ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "LUPIN.NS"],
            # ── Telecom ──────────────────────────────────────
            "Telecom Services":        ["BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS", "MTNL.NS"],
            # ── Internet ─────────────────────────────────────
            "Internet Content":        ["META", "GOOGL", "SNAP", "PINS", "RDDT"],
            "Internet Retail":         ["AMZN", "BABA", "JD", "EBAY", "ETSY", "SHOP"],
            "E-Commerce":              ["ZOMATO.NS", "NYKAA.NS", "DMART.NS", "PAYTM.NS"],
            # ── FMCG / Consumer ──────────────────────────────
            "Consumer Defensive":      ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS"],
            "Household & Personal":    ["HINDUNILVR.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "EMAMILTD.NS"],
            # ── Cement / Materials ───────────────────────────
            "Building Materials":      ["ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "DALBHARAT.NS", "JKCEMENT.NS"],
            "Steel":                   ["TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS", "HINDALCO.NS", "VEDL.NS"],
            "Metals & Mining":         ["HINDALCO.NS", "VEDL.NS", "NMDC.NS", "COALINDIA.NS", "MOIL.NS"],
        }

        # ── Keyword-based forced mapping (industry string → PEER_MAP key) ──
        # Catches cases where yfinance returns a non-standard industry label
        KEYWORD_FORCE = {
            "rail":        "Railways",
            "train":       "Railways",
            "wagon":       "Railways",
            "locomotive":  "Railways",
            "metro":       "Railways",
            "defence":     "Defence",
            "defense":     "Defence",
            "shipbuild":   "Defence",
            "ordnance":    "Defence",
            "missile":     "Defence",
            "fertiliz":    "Fertilizers",
            "agrochemi":   "Fertilizers",
            "nbfc":        "NBFC",
            "microfinan":  "NBFC",
            "power financ":"Power Finance",
            "infrastruc":  "Engineering & Construction",
            "construct":   "Engineering & Construction",
            "epc":         "Engineering & Construction",
            "bank":        "Banks—Regional",
            "electric vehicle": "Electric Vehicles",
            "ev ":         "Electric Vehicles",
            "auto manuf":  "Auto Manufacturers—US",
        }

        peers_raw = []
        industry_l = industry.lower()
        sector_l   = sector.lower()

        # Step 1: Keyword-forced match (highest priority — fixes Railway/Defence/Bank mismatches)
        forced_key = None
        for kw, map_key in KEYWORD_FORCE.items():
            if kw in industry_l:
                forced_key = map_key
                break
        # Also check sector string for bank keyword if industry didn't match
        if forced_key is None:
            for kw, map_key in KEYWORD_FORCE.items():
                if kw in sector_l:
                    forced_key = map_key
                    break
        if forced_key and forced_key in PEER_MAP:
            peers_raw = [t for t in PEER_MAP[forced_key] if t.upper() != ticker.upper()][:3]

        # Step 2: Exact industry string match against PEER_MAP keys
        if len(peers_raw) < 3:
            for key, tickers in PEER_MAP.items():
                if key.lower() in industry_l or industry_l in key.lower():
                    candidates = [t for t in tickers if t.upper() != ticker.upper() and t not in peers_raw]
                    peers_raw  = list(dict.fromkeys(peers_raw + candidates))[:3]
                    if len(peers_raw) >= 3:
                        break

        # Step 3: Sector-level match
        if len(peers_raw) < 3:
            for key, tickers in PEER_MAP.items():
                if sector_l and (sector_l in key.lower() or key.lower() in sector_l):
                    candidates = [t for t in tickers if t.upper() != ticker.upper() and t not in peers_raw]
                    peers_raw  = list(dict.fromkeys(peers_raw + candidates))[:3]
                    if len(peers_raw) >= 3:
                        break

        # Step 4: yfinance recommendations filtered by same sector
        if len(peers_raw) < 3:
            try:
                rec_df = stock.recommendations
                if rec_df is not None and not rec_df.empty and "symbol" in rec_df.columns:
                    for sym in rec_df["symbol"].dropna().unique():
                        if sym.upper() == ticker.upper() or sym in peers_raw:
                            continue
                        try:
                            peer_info = yf.Ticker(sym).info or {}
                            peer_sector = peer_info.get("sector", "")
                            if sector and peer_sector and peer_sector.lower() == sector.lower():
                                peers_raw.append(sym)
                        except Exception:
                            pass
                        if len(peers_raw) >= 3:
                            break
            except Exception:
                pass

        # Step 5: yfinance same-industry search (no cross-sector fallback)
        # Only add peers that share the same sector as the searched stock
        if len(peers_raw) < 3 and sector:
            try:
                screener_tickers = yf.Tickers(" ".join(
                    [t for lst in PEER_MAP.values() for t in lst]
                ))
                for sym, tkr_obj in screener_tickers.tickers.items():
                    if sym.upper() == ticker.upper() or sym in peers_raw:
                        continue
                    try:
                        peer_info = tkr_obj.info or {}
                        peer_sector = peer_info.get("sector", "")
                        if peer_sector and peer_sector.lower() == sector.lower():
                            peers_raw.append(sym)
                    except Exception:
                        pass
                    if len(peers_raw) >= 3:
                        break
            except Exception:
                pass

        # No cross-sector fallback — if we can't find same-sector peers, show what we have
        peers_raw = peers_raw[:3]

        all_tickers = [ticker] + peers_raw
        rows = []
        base_is_us = not (ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO"))
        usd_inr = _get_usd_inr_rate()

        for t in all_tickers:
            peer_indian = t.upper().endswith(".NS") or t.upper().endswith(".BO")
            try:
                i = yf.Ticker(t).info or {}
                pe  = i.get("trailingPE") or i.get("forwardPE")
                de  = i.get("debtToEquity")
                mc  = i.get("marketCap")
                name = i.get("shortName") or i.get("longName") or t
                peer_sector = i.get("sector", "")

                # Sector validation: skip peers whose sector doesn't match,
                # but only when both stocks are in the same region (avoid
                # dropping valid cross-regional comparisons where sector
                # labels differ between NSE and NYSE data)
                same_region = (peer_indian == base_is_us is False) or (not peer_indian and base_is_us)
                if t.upper() != ticker.upper() and sector and peer_sector and same_region:
                    if peer_sector.lower() != sector.lower():
                        continue

                mc_str = _fmt_market_cap(mc, peer_indian, base_is_us, usd_inr) if mc else "N/A"

                # D/E: yfinance returns raw ratio for US stocks, x100 for Indian stocks
                if de is not None:
                    de_val = round(float(de) / 100, 2) if peer_indian else round(float(de), 2)
                else:
                    de_val = None

                ret_1y = i.get("52WeekChange")
                if ret_1y is not None:
                    ret_1y = round(float(ret_1y) * 100, 1)

                rows.append({
                    "name":       name[:22],
                    "ticker":     t,
                    "pe":         round(float(pe), 1) if pe else None,
                    "de":         de_val,
                    "market_cap": mc_str,
                    "return_1y":  ret_1y,
                    "is_current": t.upper() == ticker.upper(),
                })
            except Exception:
                rows.append({
                    "name": t, "ticker": t,
                    "pe": None, "de": None, "market_cap": "N/A", "return_1y": None,
                    "is_current": t.upper() == ticker.upper(),
                })
        return rows
    except Exception:
        return []


def display_peer_comparison(ticker: str):
    """Render the Sector Peer Comparison table."""
    with st.expander("🏭 Sector Peer Comparison", expanded=True):
        peers = fetch_peer_comparison(ticker)
        if not peers:
            st.caption("Peer data unavailable")
            return

        th = ("background:#1e293b; color:#94a3b8; font-size:0.72rem; font-weight:700; "
              "letter-spacing:0.8px; padding:8px 12px; text-align:center; "
              "border-bottom:1px solid #334155;")
        td = ("font-size:0.82rem; padding:7px 12px; text-align:center; "
              "border-bottom:1px solid rgba(255,255,255,0.04);")

        html = (
            "<table style='width:100%; border-collapse:collapse; "
            "background:rgba(15,23,42,0.6); border-radius:10px; overflow:hidden;'>"
            f"<thead><tr>"
            f"<th style='{th} text-align:left;'>Company</th>"
            f"<th style='{th}'>Ticker</th>"
            f"<th style='{th}'>P/E Ratio</th>"
            f"<th style='{th}'>Debt / Equity</th>"
            f"<th style='{th}'>Market Cap</th>"
            f"<th style='{th}'>1Y Return</th>"
            f"</tr></thead><tbody>"
        )

        for row in peers:
            highlight = "background:rgba(99,102,241,0.12); border-left:3px solid #6366f1;" if row["is_current"] else ""
            badge = " <span style='background:#6366f1;color:#fff;font-size:0.6rem;padding:1px 5px;border-radius:4px;'>YOU</span>" if row["is_current"] else ""

            pe_val  = _safe(row["pe"],  "{:.1f}x")
            mc_val  = row.get("market_cap") or "-"

            # D/E coloring
            de = row.get("de")
            if de is not None:
                de_color = "#22c55e" if de < 0.5 else ("#f59e0b" if de < 1.5 else "#ef4444")
                de_str = f"<span style='color:{de_color}; font-weight:700;'>{de:.2f}x</span>"
            else:
                de_str = "<span style='color:#475569;'>-</span>"

            ret_1y = row.get("return_1y")
            if ret_1y is not None:
                ret_color = "#22c55e" if ret_1y >= 0 else "#ef4444"
                ret_arrow = "▲" if ret_1y >= 0 else "▼"
                ret_str = f"<span style='color:{ret_color};font-weight:700;'>{ret_arrow} {abs(ret_1y):.1f}%</span>"
            else:
                ret_str = "<span style='color:#475569;'>-</span>"

            html += (
                f"<tr style='{highlight}'>"
                f"<td style='{td} text-align:left; color:#e2e8f0; font-weight:600;'>{row['name']}{badge}</td>"
                f"<td style='{td} color:#94a3b8; font-family:monospace;'>{row['ticker']}</td>"
                f"<td style='{td} color:#e2e8f0;'>{pe_val}</td>"
                f"<td style='{td}'>{de_str}</td>"
                f"<td style='{td} color:#e2e8f0;'>{mc_val}</td>"
                f"<td style='{td}'>{ret_str}</td>"
                f"</tr>"
            )
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption("P/E = Price/Earnings · D/E = Debt/Equity (green <0.5, amber <1.5, red ≥1.5) · Market Cap: ₹Cr/Lakh Cr for Indian stocks, $B/$T for US stocks · Indian caps also show $ equivalent when comparing with US stocks · 1Y Return = 52-Week Price Change")


def compute_qvm_scores(ticker: str, recommendation: Recommendation) -> dict:
    """Compute Quality, Valuation, Momentum scores (0-100) from live data."""
    try:
        import yfinance as yf
        import numpy as np

        info = yf.Ticker(ticker).info or {}
        df   = yf.Ticker(ticker).history(period="6mo")

        roe = info.get("returnOnEquity") or 0
        de  = (info.get("debtToEquity") or 0) / 100
        pm  = info.get("profitMargins") or 0
        q_roe = min(max(roe * 100 / 25 * 100, 0), 100)
        q_de  = max(0, 100 - de * 40)
        q_pm  = min(max(pm * 100 / 20 * 100, 0), 100)
        quality = round((q_roe + q_de + q_pm) / 3, 1)

        pe = info.get("trailingPE") or info.get("forwardPE") or 0
        pb = info.get("priceToBook") or 0
        v_pe = max(0, 100 - (pe / 50 * 100)) if pe > 0 else 50
        v_pb = max(0, 100 - (pb / 5  * 100)) if pb > 0 else 50
        valuation = round((v_pe + v_pb) / 2, 1)

        momentum = 50.0
        if not df.empty and len(df) >= 50:
            close = df["Close"]
            ma50  = float(close.rolling(50).mean().iloc[-1])
            price = float(close.iloc[-1])
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, float("nan"))
            rsi   = float(100 - (100 / (1 + rs.iloc[-1])))
            m_ma  = 75 if price > ma50 else 30
            m_rsi = min(max(rsi, 0), 100)
            momentum = round((m_ma + m_rsi) / 2, 1)

        def _label(score):
            if score >= 70: return "Strong"
            if score >= 45: return "Moderate"
            return "Weak"

        return {
            "quality":   min(max(quality,   0), 100),
            "valuation": min(max(valuation, 0), 100),
            "momentum":  min(max(momentum,  0), 100),
            "q_label":   _label(quality),
            "v_label":   _label(valuation),
            "m_label":   _label(momentum),
        }
    except Exception:
        return {"quality": 50, "valuation": 50, "momentum": 50,
                "q_label": "N/A", "v_label": "N/A", "m_label": "N/A"}


def display_qvm_scores(ticker: str, recommendation: Recommendation):
    """Render visual QVM (Quality-Valuation-Momentum) progress bars."""
    with st.expander("📐 Quality · Valuation · Momentum (QVM)", expanded=True):
        qvm = compute_qvm_scores(ticker, recommendation)

        def _bar(label, score, color, sub_label, icon):
            pct = round(score, 1)
            return (
                f"<div style='margin-bottom:1.1rem;'>"
                f"<div style='display:flex; justify-content:space-between; margin-bottom:4px;'>"
                f"<span style='font-size:0.85rem; font-weight:700; color:#e2e8f0;'>{icon} {label}</span>"
                f"<span style='font-size:0.85rem; font-weight:800; color:{color};'>{pct}/100 "
                f"<span style='font-size:0.72rem; color:{color}; margin-left:4px;'>{sub_label}</span></span>"
                f"</div>"
                f"<div style='background:rgba(255,255,255,0.07); border-radius:999px; height:12px; overflow:hidden;'>"
                f"<div style='width:{pct}%; height:100%; border-radius:999px; "
                f"background:linear-gradient(90deg, {color}88, {color});'></div>"
                f"</div></div>"
            )

        q_color = "#22c55e" if qvm["quality"]   >= 70 else ("#f59e0b" if qvm["quality"]   >= 45 else "#ef4444")
        v_color = "#22c55e" if qvm["valuation"] >= 70 else ("#f59e0b" if qvm["valuation"] >= 45 else "#ef4444")
        m_color = "#22c55e" if qvm["momentum"]  >= 70 else ("#f59e0b" if qvm["momentum"]  >= 45 else "#ef4444")

        bars_html = (
            _bar("Quality",   qvm["quality"],   q_color, qvm["q_label"], "🏆") +
            _bar("Valuation", qvm["valuation"], v_color, qvm["v_label"], "💰") +
            _bar("Momentum",  qvm["momentum"],  m_color, qvm["m_label"], "🚀")
        )
        composite = round((qvm["quality"] + qvm["valuation"] + qvm["momentum"]) / 3, 1)
        comp_color = "#22c55e" if composite >= 65 else ("#f59e0b" if composite >= 45 else "#ef4444")

        st.markdown(
            f"<div style='background:rgba(15,23,42,0.85); border:1px solid rgba(255,255,255,0.07); "
            f"border-radius:14px; padding:1.2rem 1.4rem;'>"
            f"{bars_html}"
            f"<div style='border-top:1px solid rgba(255,255,255,0.07); padding-top:0.8rem; "
            f"display:flex; justify-content:space-between; align-items:center;'>"
            f"<span style='font-size:0.82rem; color:#94a3b8;'>Composite QVM Score</span>"
            f"<span style='font-size:1.4rem; font-weight:900; color:{comp_color};'>{composite}/100</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        st.caption("Quality = ROE + Margins + Debt · Valuation = P/E + P/B · Momentum = Price vs MA50 + RSI")


def display_ai_reasoning(recommendation: Recommendation):
    with st.expander("🤖 AI Analysis & Reasoning", expanded=True):
        st.markdown("### Investment Summary")
        reasoning = recommendation.reasoning if recommendation.reasoning else "Calculated data not available"
        st.write(reasoning)

        # ── News conflict warning injected into reasoning ─────
        conflict = st.session_state.get("news_conflict")
        if conflict:
            st.markdown(
                f"<div style='background:rgba(245,158,11,0.10); border:1px solid #f59e0b55; "
                f"border-left:4px solid #f59e0b; border-radius:8px; "
                f"padding:0.7rem 1rem; margin-top:0.8rem;'>"
                f"<b style='color:#f59e0b;'>⚠️ AI Warning — Sentiment Conflict</b><br>"
                f"<span style='font-size:0.85rem; color:#cbd5e1;'>"
                f"The technical signal is <b>{conflict['signal']}</b>, but current news sentiment "
                f"is <b style='color:#f59e0b;'>{conflict['direction']}</b> "
                f"(AI score: <b>{conflict['ai_score']:+.2f}</b>). "
                f"Confidence has been adjusted to <b style='color:#f59e0b;'>{conflict['adj_conf']}</b>. "
                f"Consider waiting for news sentiment to align before acting."
                f"</span>"
                f"{'<br><span style=\"font-size:0.8rem; color:#94a3b8; margin-top:4px; display:block;\">🤖 ' + conflict['insight'] + '</span>' if conflict.get('insight') else ''}"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.caption(f"Analysis generated at: {format_ist_timestamp(recommendation.timestamp)}")
        st.caption("Powered by Anthropic Claude 3")


def send_email(receiver_email: str, pdf_file_path: bytes, ticker: str) -> None:
    """
    Send a PDF report as an email attachment via Gmail SMTP (port 587 + STARTTLS).
    Credentials are read from st.secrets first, then os.getenv as fallback.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    # Prefer st.secrets (Streamlit Cloud), fall back to .env / environment variables
    try:
        sender_email = st.secrets.get("SENDER_EMAIL", "") or os.getenv("SENDER_EMAIL", "")
        app_password = st.secrets.get("APP_PASSWORD", "") or os.getenv("APP_PASSWORD", "")
    except Exception:
        sender_email = os.getenv("SENDER_EMAIL", "")
        app_password = os.getenv("APP_PASSWORD", "")

    # Strip any accidental spaces from the password (common copy-paste issue)
    app_password = app_password.replace(' ', '')

    if not sender_email:
        raise ValueError(
            "SENDER_EMAIL is not configured. "
            "Add it to .streamlit/secrets.toml or your .env file."
        )
    if not app_password:
        raise ValueError(
            "APP_PASSWORD is not configured. "
            "Add your Gmail App Password to .streamlit/secrets.toml or your .env file."
        )

    print(f"Attempting to send email from: {sender_email}")

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = f"Research Analysis Report — {ticker} | Lakshya AI"

    body = (
        f"Dear Investor,\n\n"
        f"Please find the attached Research Analysis Report for {ticker} "
        f"by Lakshya AI Financial Intelligence.\n\n"
        f"This report includes technical indicators, fundamental analysis, sentiment analysis, "
        f"and AI-generated investment insights.\n\n"
        f"Disclaimer: This analysis is for educational purposes only. "
        f"Please consult a SEBI-registered financial advisor before making any investment decisions.\n\n"
        f"Best regards,\nLakshya AI Financial Intelligence"
    )
    msg.attach(MIMEText(body, "plain"))

    part = MIMEBase("application", "octet-stream")
    part.set_payload(pdf_file_path)
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{ticker}_Research_Report.pdf"',
    )
    msg.attach(part)

    # Port 587 + STARTTLS (more reliable than SSL/465 for most networks)
    # Raises on failure — caller is responsible for displaying messages
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())


def generate_pdf_report(recommendation: Recommendation) -> bytes:
    """
    Professional RA-grade Equity Research PDF.
    Includes Target Price, Stop Loss, Risk-Reward Ratio, and SEBI Legal Disclosure.
    All text is sanitised to latin-1 before rendering.
    """
    from fpdf import FPDF
    import yfinance as yf

    DISCLAIMER = "SEBI Research Analyst Regulations, 2014. For educational purposes only."

    sig_colors = {
        "BUY":  (34, 197, 94),
        "SELL": (239, 68, 68),
        "HOLD": (245, 158, 11),
    }
    sr, sg, sb = sig_colors.get(recommendation.signal, (100, 116, 139))

    # ── Compute Target Price, Stop Loss, Risk-Reward ──────────
    try:
        import numpy as np
        info       = yf.Ticker(recommendation.ticker).info or {}
        curr_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        signal     = recommendation.signal

        # Try Monte Carlo P90 as the primary target price source
        mc_target = None
        try:
            _df = yf.Ticker(recommendation.ticker).history(period="6mo")
            if not _df.empty:
                _mc = compute_monte_carlo_forecast(_df, days=15, simulations=500)
                mc_target = _mc.get("p90_target")
                mc_prob   = _mc.get("prob_of_profit", 0)
        except Exception:
            mc_prob = 0

        if signal == "BUY" and curr_price:
            if mc_target and mc_target > curr_price * 1.05:
                target_price = round(float(mc_target), 2)
            else:
                analyst_target = info.get("targetMeanPrice") or info.get("targetHighPrice")
                if analyst_target and float(analyst_target) > curr_price * 1.05:
                    target_price = round(float(analyst_target), 2)
                else:
                    target_price = round(curr_price * 1.175, 2)
        elif signal == "SELL" and curr_price:
            analyst_target = info.get("targetLowPrice") or info.get("targetMeanPrice")
            if analyst_target and float(analyst_target) < curr_price * 0.95:
                target_price = round(float(analyst_target), 2)
            else:
                target_price = round(curr_price * 0.85, 2)
        else:
            target_raw = info.get("targetMeanPrice")
            target_price = round(float(target_raw), 2) if target_raw else (
                round(curr_price * 1.05, 2) if curr_price else None
            )

        stop_loss = round(curr_price * 0.92, 2) if curr_price else None

        if target_price and stop_loss and curr_price and (curr_price - stop_loss) > 0:
            rr_ratio = round((target_price - curr_price) / (curr_price - stop_loss), 2)
        else:
            rr_ratio = None

        currency = get_currency_symbol(recommendation.ticker)
        tp_str   = f"{currency}{target_price:,.2f}" if target_price else "-"
        sl_str   = f"{currency}{stop_loss:,.2f}"    if stop_loss   else "-"
        rr_str   = f"{rr_ratio:.2f}:1"              if rr_ratio    else "-"
        cp_str   = f"{currency}{curr_price:,.2f}"   if curr_price  else "-"
        prob_str = f"{mc_prob:.1f}%" if mc_prob else "-"
    except Exception:
        tp_str = sl_str = rr_str = cp_str = prob_str = "-"

    class ResearchPDF(FPDF):
        def __init__(self):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.set_margins(18, 28, 18)
            self.set_auto_page_break(auto=True, margin=20)

        def header(self):
            self.set_fill_color(10, 20, 50)
            self.rect(0, 0, 210, 16, style="F")
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(255, 255, 255)
            self.set_xy(18, 4)
            self.cell(90, 7, "LAKSHYA AI  |  RESEARCH ANALYST REPORT", align="L")
            self.set_font("Helvetica", "", 8)
            self.set_text_color(148, 163, 184)
            self.set_xy(100, 4)
            self.cell(92, 7, _sanitise_pdf(recommendation.ticker) + "  |  " +
                      __import__('datetime').datetime.now().strftime("%d %b %Y"), align="R")
            self.set_draw_color(37, 99, 235)
            self.set_line_width(0.8)
            self.line(0, 16, 210, 16)
            self.ln(4)

        def footer(self):
            self.set_y(-13)
            self.set_draw_color(200, 210, 220)
            self.set_line_width(0.3)
            self.line(18, self.get_y(), 192, self.get_y())
            self.ln(1)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(120, 130, 140)
            self.cell(160, 5, DISCLAIMER, align="L")
            self.cell(0, 5, f"Page {self.page_no()}", align="R")

    pdf = ResearchPDF()
    pdf.add_page()

    # ── helpers ───────────────────────────────────────────────
    def _fmt(val) -> str:
        if val is None:
            return "-"
        s = str(val).strip()
        return s if s and s.lower() not in ("none", "nan") else "-"

    def h1(txt):
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(10, 20, 50)
        pdf.cell(0, 10, _sanitise_pdf(txt), new_x="LMARGIN", new_y="NEXT")

    def h2(txt):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 7, _sanitise_pdf(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(37, 99, 235)
        pdf.set_line_width(0.35)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 174, pdf.get_y())
        pdf.ln(3)
        pdf.set_text_color(15, 23, 42)

    def body(txt):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(0, 5.5, _sanitise_pdf(_fmt(txt)))
        pdf.ln(2)

    def kv_table(rows):
        col = pdf.epw / 2
        for i, (key, val) in enumerate(rows):
            pdf.set_fill_color(241, 245, 249) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(col, 7, _sanitise_pdf(_fmt(key)), border="B", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(col, 7, _sanitise_pdf(_fmt(val)), border="B", fill=True,
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── PAGE 1 ────────────────────────────────────────────────
    h1(f"EQUITY RESEARCH REPORT: {recommendation.ticker}")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "Generated by Lakshya AI Financial Intelligence",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(10, 20, 50)
    pdf.set_line_width(1.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 174, pdf.get_y())
    pdf.ln(6)

    # ── Summary box ───────────────────────────────────────────
    box_x, box_y = pdf.l_margin, pdf.get_y()
    box_w, box_h = pdf.epw, 32
    pdf.set_fill_color(10, 20, 50)
    pdf.rect(box_x, box_y, box_w, box_h, style="F")
    pdf.set_fill_color(sr, sg, sb)
    pdf.rect(box_x, box_y, 5, box_h, style="F")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(sr, sg, sb)
    pdf.set_xy(box_x + 9, box_y + 4)
    pdf.cell(55, 12, recommendation.signal)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(200, 210, 220)
    pdf.set_xy(box_x + 9, box_y + 18)
    pdf.cell(55, 6, f"Confidence: {recommendation.confidence_score}/100")

    bx, by, bw, bh = box_x + 68, box_y + 8, box_w - 78, 5
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(bx, by, bw, bh, style="F")
    pdf.set_fill_color(sr, sg, sb)
    pdf.rect(bx, by, bw * recommendation.confidence_score / 100, bh, style="F")

    tech = parse_pipe_summary(recommendation.technical_summary or "")
    metrics = [
        ("Price",    tech.get("Price",  cp_str)),
        ("MA-50",    tech.get("MA-50",  "N/A")),
        ("MA-200",   tech.get("MA-200", "N/A")),
        ("RSI",      tech.get("RSI",    "N/A")),
    ]
    cell_w = bw / 4
    for j, (mk, mv) in enumerate(metrics):
        cx = bx + j * cell_w
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(148, 163, 184)
        pdf.set_xy(cx, box_y + 16)
        pdf.cell(cell_w, 4, _sanitise_pdf(mk), align="C")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(cx, box_y + 22)
        pdf.cell(cell_w, 5, _sanitise_pdf(_fmt(mv)), align="C")

    pdf.set_y(box_y + box_h + 6)

    # ── RA Price Targets (NEW) ────────────────────────────────
    h2("Price Targets & Risk Parameters (6-12 Month Horizon)")
    kv_table([
        ("Current Market Price",  cp_str),
        ("Target Price (P90 MC)", tp_str),
        ("Stop Loss",             sl_str),
        ("Risk-Reward Ratio",     rr_str),
        ("Probability of Profit", prob_str),
        ("Investment Horizon",    "6 - 12 Months"),
    ])

    # ── Technical Outlook ─────────────────────────────────────
    h2("Technical Outlook")
    kv_table([
        ("Current Price",  tech.get("Price",   cp_str)),
        ("50-Day MA",      tech.get("MA-50",   "N/A")),
        ("200-Day MA",     tech.get("MA-200",  "N/A")),
        ("RSI (14-day)",   tech.get("RSI",     "N/A")),
        ("Signals",        tech.get("Signals", "N/A")),
    ])
    body(recommendation.technical_summary or "Technical data not available")

    # ── Fundamental Snapshot ──────────────────────────────────
    h2("Fundamental Snapshot")
    fund = parse_pipe_summary(recommendation.fundamental_summary or "")

    # Earnings trend: use pipe summary first; if N/A, derive from annual net income
    earnings_trend = fund.get("Earnings Trend", "N/A")
    if not earnings_trend or earnings_trend.strip() in ("N/A", "-", ""):
        try:
            _is_indian = recommendation.ticker.upper().endswith(".NS") or recommendation.ticker.upper().endswith(".BO")
            _af = yf.Ticker(recommendation.ticker).financials
            if _af is not None and not _af.empty and "Net Income" in _af.index:
                _ni = _af.loc["Net Income"].dropna()
                if len(_ni) >= 2:
                    _latest = float(_ni.iloc[0])
                    _prev   = float(_ni.iloc[1])
                    _trend  = "Growing" if _latest > _prev else "Declining"
                    _ni_str = _fmt_inr(abs(_latest), _is_indian)
                    _pct    = (_latest - _prev) / abs(_prev) * 100 if _prev != 0 else 0
                    earnings_trend = f"{_trend} ({_pct:+.1f}% YoY, Net Income: {_ni_str})"
        except Exception:
            pass

    kv_table([
        ("Market Cap",     fund.get("Market Cap",     "N/A")),
        ("P/E Ratio",      fund.get("P/E",            "N/A")),
        ("Earnings Trend", earnings_trend),
        ("Valuation",      fund.get("Valuation",      "N/A")),
    ])
    body(recommendation.fundamental_summary or "Fundamental data not available")

    # ── PAGE 2 ────────────────────────────────────────────────
    pdf.add_page()

    h2("Market Sentiment")
    body(recommendation.sentiment_summary or "Sentiment data not available")
    pdf.ln(3)

    h2("Investment Rationale")
    body(recommendation.reasoning or "Reasoning not available")
    pdf.ln(3)

    # ── Specific Risks (SEBI Compliance) ─────────────────────
    h2("Specific Risks")
    is_indian_ticker = recommendation.ticker.upper().endswith(".NS") or recommendation.ticker.upper().endswith(".BO")
    try:
        _info = yf.Ticker(recommendation.ticker).info or {}
        _sector = _info.get("sector", "")
        _beta   = _info.get("beta")
        _de     = _info.get("debtToEquity")
    except Exception:
        _sector, _beta, _de = "", None, None

    risk_items = []

    # Market / volatility risk
    if _beta is not None:
        beta_val = round(float(_beta), 2)
        if beta_val > 1.5:
            risk_items.append(f"High Volatility Risk: Beta of {beta_val} indicates significantly higher price swings than the broader market.")
        elif beta_val > 1.0:
            risk_items.append(f"Moderate Volatility Risk: Beta of {beta_val} suggests above-market price sensitivity.")
        else:
            risk_items.append(f"Low Volatility Risk: Beta of {beta_val} indicates relatively stable price movement.")

    # Leverage / debt risk
    if _de is not None:
        de_val = round(float(_de) / 100, 2)
        if de_val > 1.5:
            risk_items.append(f"High Leverage Risk: Debt-to-Equity ratio of {de_val:.2f} may strain cash flows in rising interest rate environments.")
        elif de_val > 0.5:
            risk_items.append(f"Moderate Leverage Risk: Debt-to-Equity ratio of {de_val:.2f} warrants monitoring of debt servicing capacity.")

    # Sector-specific risks
    sector_risks = {
        "Technology":        "Sector Risk: Technology stocks are susceptible to rapid disruption, regulatory scrutiny on data privacy, and valuation compression during rate hikes.",
        "Financial Services": "Sector Risk: Banking and financial stocks face credit cycle risk, NPA (Non-Performing Asset) exposure, and sensitivity to RBI monetary policy changes.",
        "Energy":            "Sector Risk: Energy sector is exposed to commodity price volatility, geopolitical supply disruptions, and transition risk from renewable energy adoption.",
        "Healthcare":        "Sector Risk: Pharmaceutical stocks carry drug approval risk (USFDA/CDSCO), pricing pressure, and patent cliff exposure.",
        "Consumer Cyclical": "Sector Risk: Consumer discretionary stocks are sensitive to economic slowdowns, inflation eroding purchasing power, and rural demand fluctuations.",
        "Real Estate":       "Sector Risk: Real estate stocks are highly sensitive to interest rate cycles, regulatory changes (RERA), and liquidity conditions.",
        "Utilities":         "Sector Risk: Utility companies face regulatory tariff risk, capex-heavy balance sheets, and exposure to fuel cost pass-through delays.",
        "Industrials":       "Sector Risk: Industrial companies are exposed to order book cyclicality, raw material cost inflation, and government capex policy changes.",
    }
    for key, risk_text in sector_risks.items():
        if _sector and key.lower() in _sector.lower():
            risk_items.append(risk_text)
            break

    # Indian market-specific risks
    if is_indian_ticker:
        risk_items.append("Currency & Macro Risk: INR depreciation, FII outflows, and RBI policy rate changes can materially impact stock valuations.")
        risk_items.append("Regulatory Risk: SEBI regulations, changes in tax policy (LTCG/STCG), and sector-specific government interventions may affect returns.")

    # Liquidity risk
    risk_items.append("Liquidity Risk: Investors should assess average daily traded volume before sizing positions. Illiquid stocks may have wider bid-ask spreads.")

    # General market risk
    risk_items.append("General Market Risk: Equity investments are subject to market risk. Past performance is not a guarantee of future returns. Investors may lose part or all of their invested capital.")

    if risk_items:
        for item in risk_items:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(0, 5, _sanitise_pdf(f"  - {item}"))
            pdf.ln(1)
    else:
        body("Standard market, liquidity, and regulatory risks apply. Please consult a SEBI-registered financial advisor.")
    pdf.ln(3)

    # ── Sector Peer Comparison ────────────────────────────────
    h2("Sector Peer Comparison")
    peers = fetch_peer_comparison(recommendation.ticker)
    if peers:
        col_w = pdf.epw / 5
        headers = ["Company", "Ticker", "P/E", "Mkt Cap", "1Y Return"]
        pdf.set_fill_color(30, 41, 59)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(148, 163, 184)
        for h_txt in headers:
            pdf.cell(col_w, 6, _sanitise_pdf(h_txt), border="B", fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for idx_p, p in enumerate(peers):
            pdf.set_fill_color(241, 245, 249) if idx_p % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(15, 23, 42)
            ret_str = f"{p['return_1y']:+.1f}%" if p.get("return_1y") is not None else "-"
            row_vals = [
                p["name"][:18],
                p["ticker"],
                f"{p['pe']:.1f}x" if p["pe"] else "-",
                p.get("market_cap") or "-",
                ret_str,
            ]
            for v in row_vals:
                pdf.cell(col_w, 6, _sanitise_pdf(str(v)), border="B", fill=True, align="C")
            pdf.ln()
        pdf.ln(3)
    else:
        body("Peer comparison data unavailable")
    pdf.ln(2)

    pdf.set_draw_color(200, 210, 220)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 174, pdf.get_y())
    pdf.ln(4)
    kv_table([
        ("Generated at", format_ist_timestamp(recommendation.timestamp)),
        ("Powered by",   "Anthropic Claude 3 via AWS Bedrock"),
        ("Platform",     "Lakshya AI - Multi-Agent Stock Analysis"),
    ])

    # ── PAGE 3: SEBI Legal Disclosure & Disclaimer (NEW) ─────
    pdf.add_page()

    h2("Legal Disclosure & Disclaimer (SEBI RA Regulations, 2014)")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(10, 20, 50)
    pdf.cell(0, 7, "Research Analyst Registration Details", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    kv_table([
        ("RA Name / Entity",        "[Your Name / Firm Name]"),
        ("SEBI RA Registration No.", "[Enter SEBI Registration Number e.g. INH000XXXXXX]"),
        ("Contact Email",            "[Your Email Address]"),
        ("Contact Phone",            "[Your Phone Number]"),
        ("Registered Address",       "[Your Registered Office Address]"),
        ("Report Date",              __import__('datetime').datetime.now().strftime("%d %b %Y")),
        ("Analyst Name",             "[Analyst Name]"),
    ])

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(10, 20, 50)
    pdf.cell(0, 7, "Conflict of Interest Declaration", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    conflict_text = (
        "1. The Research Analyst or its associates do not have any financial interest in the subject company. "
        "2. The Research Analyst or its associates do not hold any actual/beneficial ownership of 1% or more "
        "in the securities of the subject company. "
        "3. The Research Analyst or its associates have not received any compensation from the subject company "
        "in the past twelve months. "
        "4. The Research Analyst has not served as an officer, director, or employee of the subject company. "
        "5. The Research Analyst is not engaged in market making activity for the subject company."
    )
    body(conflict_text)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(10, 20, 50)
    pdf.cell(0, 7, "General Disclaimer", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    disclaimer_text = (
        "This report has been prepared by Lakshya AI for informational and educational purposes only. "
        "It does not constitute investment advice, a solicitation, or an offer to buy or sell any securities. "
        "The information contained herein is based on sources believed to be reliable, but no representation "
        "or warranty, express or implied, is made as to its accuracy, completeness, or timeliness. "
        "Past performance is not indicative of future results. Investments in securities are subject to "
        "market risks. Please read all scheme-related documents carefully before investing. "
        "The target price and stop loss mentioned are indicative and based on algorithmic analysis. "
        "Investors should consult a SEBI-registered financial advisor before making any investment decision. "
        "Lakshya AI and its affiliates accept no liability for any loss arising from the use of this report."
    )
    body(disclaimer_text)

    return bytes(pdf.output())


def main():
    # ── Single persistent hero header ─────────────────────────
    st.markdown("""
        <div style="
            text-align: center;
            padding: 2.5rem 1rem 0.8rem;
            user-select: none;
            pointer-events: none;
        ">
            <h1 style="
                font-size: 45px;
                font-weight: 900;
                line-height: 1.2;
                margin: 0 0 0.4rem 0;
                letter-spacing: -0.5px;
            ">
                <span style="font-style: normal;">🎯</span>
                <span style="
                    background: linear-gradient(90deg, #0066CC 0%, #00A8E8 50%, #00D4FF 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">&nbsp;Lakshya AI &mdash; Stock Market Analysis</span>
            </h1>
            <p style="
                font-size: 1rem;
                color: #888;
                font-style: italic;
                margin: 0 0 1.2rem 0;
                letter-spacing: 0.3px;
            ">(AI-Powered Multi-Agent Stock Analysis System)</p>
            <hr style="border:none; border-top:1px solid #1e293b; margin: 0 auto; width:55%;"/>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        # ── Brand Header ──────────────────────────────────────
        st.markdown(
            "<div style='text-align:center; padding: 0.6rem 0 0.4rem;'>"
            "<span style='font-size:2rem;'>🎯</span>&nbsp;"
            "<span style='font-size:1.25rem; font-weight:800; "
            "background:linear-gradient(135deg,#3b82f6,#06b6d4);"
            "-webkit-background-clip:text; -webkit-text-fill-color:transparent;'>"
            "Lakshya AI</span>"
            "</div>",
            unsafe_allow_html=True
        )

        # ── Market Status (IST-based) ──────────────────────────
        from datetime import timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(IST)
        weekday = now_ist.weekday()          # 0=Mon … 6=Sun
        hour    = now_ist.hour
        minute  = now_ist.minute
        # NSE/BSE: Mon–Fri 09:15–15:30 IST
        market_open = (weekday < 5) and (
            (hour == 9 and minute >= 15) or
            (10 <= hour <= 14) or
            (hour == 15 and minute <= 30)
        )
        if market_open:
            mkt_color, mkt_icon, mkt_label = "#22c55e", "🟢", "Market Open"
        elif weekday >= 5:
            mkt_color, mkt_icon, mkt_label = "#64748b", "⚫", "Weekend — Closed"
        else:
            mkt_color, mkt_icon, mkt_label = "#ef4444", "🔴", "Market Closed"

        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);"
            f"border-radius:8px; padding:0.4rem 0.8rem; margin:0.4rem 0 0.2rem; "
            f"display:flex; justify-content:space-between; align-items:center;'>"
            f"<span style='font-size:0.75rem; color:#94a3b8;'>NSE / BSE</span>"
            f"<span style='font-size:0.78rem; font-weight:700; color:{mkt_color};'>"
            f"{mkt_icon} {mkt_label}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # ── Index Snapshot (NIFTY 50, BANK NIFTY, SENSEX) ─────
        @st.cache_data(ttl=300)   # refresh every 5 min
        def _fetch_indices():
            try:
                import yfinance as yf
                tickers = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
                result = {}
                for name, sym in tickers.items():
                    info = yf.Ticker(sym).fast_info
                    price = getattr(info, "last_price", None)
                    prev  = getattr(info, "previous_close", None)
                    if price and prev and prev > 0:
                        chg = (price - prev) / prev * 100
                        result[name] = (price, chg)
                return result
            except Exception:
                return {}

        indices = _fetch_indices()
        if indices:
            idx_html = "<div style='margin:0.3rem 0 0.1rem;'>"
            for name, (price, chg) in indices.items():
                c = "#22c55e" if chg >= 0 else "#ef4444"
                arrow = "▲" if chg >= 0 else "▼"
                idx_html += (
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:0.22rem 0.8rem; border-bottom:1px solid rgba(255,255,255,0.04);'>"
                    f"<span style='font-size:0.72rem; color:#94a3b8;'>{name}</span>"
                    f"<span style='font-size:0.72rem; font-weight:700; color:{c};'>"
                    f"{arrow} {chg:+.2f}%</span>"
                    f"</div>"
                )
            idx_html += "</div>"
            st.markdown(idx_html, unsafe_allow_html=True)

        st.divider()

        # ── Ticker Input ───────────────────────────────────────
        raw_input = st.text_input(
            "Enter Stock Ticker Symbol",
            placeholder="e.g., RELIANCE.NS, INFY.NS, AAPL",
            help="NSE stocks: SYMBOL.NS  |  BSE: SYMBOL.BO  |  US: SYMBOL only",
        ).strip().upper()

        st.caption(
            "Don't know the ticker? "
            "[Find it on Yahoo Finance](https://finance.yahoo.com/lookup)"
        )

        st.divider()
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
        st.divider()

        # ── Risk Meter ─────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.82rem; font-weight:700; color:#cbd5e1; margin-bottom:0.4rem;'>"
            "⚠️ Risk Meter</p>",
            unsafe_allow_html=True
        )
        # Derive risk level from session state if analysis done, else show default
        _risk_level = st.session_state.get("risk_level", "Medium")
        _risk_cfg = {
            "Low":    ("#22c55e", 1, "rgba(34,197,94,0.15)"),
            "Medium": ("#f59e0b", 2, "rgba(245,158,11,0.15)"),
            "High":   ("#ef4444", 3, "rgba(239,68,68,0.15)"),
        }
        _rc, _ri, _rbg = _risk_cfg.get(_risk_level, _risk_cfg["Medium"])
        _risk_html = (
            f"<div style='background:{_rbg}; border:1px solid {_rc}44; border-radius:10px; "
            f"padding:0.5rem 0.8rem; margin-bottom:0.3rem;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;'>"
            f"<span style='font-size:0.78rem; color:#94a3b8;'>Risk Level</span>"
            f"<span style='font-size:0.9rem; font-weight:800; color:{_rc};'>{_risk_level}</span>"
            f"</div>"
            f"<div style='display:flex; gap:4px;'>"
        )
        for _seg in range(3):
            _filled = _seg < _ri
            _seg_c = _rc if _filled else "rgba(255,255,255,0.08)"
            _risk_html += f"<div style='flex:1; height:8px; border-radius:4px; background:{_seg_c};'></div>"
        _risk_html += (
            f"</div>"
            f"<div style='display:flex; justify-content:space-between; margin-top:3px;'>"
            f"<span style='font-size:0.62rem; color:#475569;'>Low</span>"
            f"<span style='font-size:0.62rem; color:#475569;'>Medium</span>"
            f"<span style='font-size:0.62rem; color:#475569;'>High</span>"
            f"</div></div>"
        )
        st.markdown(_risk_html, unsafe_allow_html=True)
        st.divider()

        # ── About ──────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.82rem; font-weight:700; color:#cbd5e1;'>"
            "ℹ️ About Lakshya AI</p>",
            unsafe_allow_html=True
        )
        agents = [
            ("📊", "Technical Agent",   "Price · MA · RSI"),
            ("💼", "Fundamental Agent", "P/E · Market Cap · Earnings"),
            ("💭", "Sentiment Agent",   "News · Market mood"),
            ("🛡️", "Risk Manager",      "Final recommendation"),
        ]
        for icon, name, desc in agents:
            st.markdown(
                f"<div style='background:#1e293b; border:1px solid #334155; "
                f"border-radius:8px; padding:0.45rem 0.7rem; margin-bottom:0.4rem;'>"
                f"<span style='font-size:1rem;'>{icon}</span> "
                f"<span style='color:#e2e8f0; font-size:0.82rem; font-weight:600;'>{name}</span><br>"
                f"<span style='color:#64748b; font-size:0.75rem;'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ── Advanced Settings (collapsed) ─────────────────────
        with st.expander("⚙️ Advanced Settings"):
            aws_profile = st.text_input(
                "AWS Profile (Optional)",
                value=os.getenv("AWS_PROFILE", ""),
                help="AWS config profile name"
            )

    # Resolve ticker (company name -> symbol, auto-suffix, etc.)
    ticker, _resolve_msg = resolve_ticker(raw_input) if raw_input else ("", None)
    if analyze_button:
        if not ticker:
            st.error("⚠️ Please enter a stock ticker symbol")
            return

        profile_name = aws_profile.strip() if aws_profile.strip() else None

        with st.spinner(f"🔄 Analyzing {ticker}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                st.session_state["analysis_complete"] = False
                st.session_state.pop("email_status", None)  # clear stale email status on new analysis
                status_text.text("Initializing system components...")
                progress_bar.progress(10)
                time.sleep(0.3)

                status_text.text("Fetching stock data...")
                progress_bar.progress(30)

                status_text.text("Running Technical Agent...")
                progress_bar.progress(45)

                status_text.text("Running Fundamental Agent...")
                progress_bar.progress(60)

                status_text.text("Running Sentiment Agent...")
                progress_bar.progress(75)

                status_text.text("Synthesizing recommendation...")
                progress_bar.progress(90)

                recommendation = analyze_stock(ticker, profile_name=profile_name)

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.3)
                progress_bar.empty()
                status_text.empty()

                st.session_state["analysis_complete"] = True

                # Derive risk level from signal + confidence for the sidebar Risk Meter
                _conf = recommendation.confidence_score or 50
                _sig  = recommendation.signal
                if _sig == "BUY" and _conf >= 70:
                    st.session_state["risk_level"] = "Low"
                elif _sig == "SELL" or _conf < 45:
                    st.session_state["risk_level"] = "High"
                else:
                    st.session_state["risk_level"] = "Medium"

                company_name = get_company_name(ticker)
                st.markdown(
                    f"<div style='text-align:center; padding:0.5rem 0;'>"
                    f"<span style='font-size:1.4rem; font-weight:700; color:#e2e8f0;'>"
                    f"{company_name}</span><br>"
                    f"<span style='font-size:0.85rem; color:#64748b;'>{ticker}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                _ = display_recommendation_card(recommendation)

                # ── Full-width chart at the top ────────────────
                _ = display_full_width_chart(recommendation)

                # ── Technical metrics row ──────────────────────
                _ = display_technical_data(recommendation)

                # ── Fundamental + Sentiment side by side ──────
                col1, col2 = st.columns(2)
                with col1:
                    _ = display_fundamental_data(recommendation)
                with col2:
                    _ = display_sentiment_data(recommendation)

                _ = display_ai_reasoning(recommendation)

                # ── QVM Scores + Peer Comparison ──────────────
                qvm_col, peer_col = st.columns(2)
                with qvm_col:
                    _ = display_qvm_scores(recommendation.ticker, recommendation)
                with peer_col:
                    _ = display_peer_comparison(recommendation.ticker)

                # ── Download buttons (prominent, full-width) ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align:center; font-size:0.9rem; color:#94a3b8; margin-bottom:0.5rem;'>"
                    "📥 Export your analysis report</p>",
                    unsafe_allow_html=True
                )
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    try:
                        pdf_bytes = generate_pdf_report(recommendation)
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary",
                            disabled=not st.session_state.get("analysis_complete", False),
                        )
                    except Exception as _pdf_err:
                        st.warning(f"PDF unavailable: {_pdf_err}")
                        st.caption("Run: `pip install fpdf2`")
                with dl_col2:
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=recommendation.to_json(),
                        file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                # ── Email Report ──────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📧 Send Report via Email", expanded=False):
                    st.markdown(
                        "<p style='font-size:0.9rem; color:#94a3b8; margin-bottom:0.5rem;'>"
                        "Send this report directly to a client's inbox</p>",
                        unsafe_allow_html=True
                    )
                    client_email = st.text_input(
                        "Client Email Address",
                        placeholder="investor@example.com",
                        key="client_email_input",
                    )
                    send_clicked = st.button(
                        "📧 Send Report",
                        use_container_width=True,
                        disabled=not st.session_state.get("analysis_complete", False),
                        key="send_email_btn",
                    )
                    if send_clicked:
                        if not client_email or "@" not in client_email:
                            st.session_state["email_status"] = ("error", "Please enter a valid email address.")
                        else:
                            with st.spinner("Generating PDF and sending email..."):
                                try:
                                    pdf_bytes_email = generate_pdf_report(recommendation)
                                    send_email(client_email, pdf_bytes_email, ticker)
                                    st.session_state["email_status"] = ("success", f"✅ Report sent successfully to {client_email}")
                                except ValueError as _ve:
                                    st.session_state["email_status"] = ("error", f"⚙️ Configuration error: {_ve}")
                                except Exception as _mail_err:
                                    import traceback
                                    print("=== EMAIL ERROR TRACEBACK ===")
                                    traceback.print_exc()
                                    print("=============================")
                                    st.session_state["email_status"] = ("error", f"❌ Failed to send email: {_mail_err}")

                    # Persistent status — survives reruns via session_state
                    _email_status = st.session_state.get("email_status")
                    if _email_status:
                        if _email_status[0] == "success":
                            st.success(_email_status[1])
                        else:
                            st.error(_email_status[1])
                st.markdown("---")

            except InvalidTickerError as e:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ **'{ticker}'** — Invalid Ticker")
                st.warning(
                    "Use a valid ticker (e.g. **TATAPOWER.NS**, **MAHSEAMLESS.NS**, **RELIANCE.NS**)\n\n"
                    "US stocks: **AAPL**, **MSFT**, **TSLA** — no suffix needed."
                )

            except AuthenticationError as e:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ Authentication Error: {str(e)}")
                st.warning("Check AWS credentials, Bedrock access, and region.")

            except DataUnavailableError as e:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ Data Unavailable: {str(e)}")
                st.info("The stock data service may be temporarily unavailable.")

            except LakshyaAIError as e:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ Analysis Error: {str(e)}")

            except Exception as e:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ Unexpected Error: {str(e)}")

    else:
        st.markdown("---")
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown(
                "<p style='text-align:center; color:#64748b; font-size:0.9rem; margin-bottom:1rem;'>"
                "How to Use"
                "</p>",
                unsafe_allow_html=True
            )

            steps = [
                ("1", "Enter <strong>Stock Ticker</strong> in Sidebar",
                 "e.g. <code>AAPL</code>, <code>TCS.NS</code>, <code>MSFT</code>"),
                ("2", "Click <strong>\"Analyze Stock\"</strong> Button",
                 "AI Agents will collect and analyze data"),
                ("3", "View <strong>Detailed Report</strong>",
                 "Technical · Fundamental · Sentiment · AI Reasoning"),
            ]
            for num, title, desc in steps:
                st.markdown(f"""
                    <div class="step-card">
                        <div class="step-number">{num}</div>
                        <div class="step-text">{title}<br>
                            <span style="color:#64748b; font-size:0.85rem;">{desc}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center; color:#475569; font-size:0.82rem;'>Powered by</p>",
                unsafe_allow_html=True
            )
            badges = ["🤖 Technical Agent", "💼 Fundamental Agent",
                      "💭 Sentiment Agent", "🛡️ Risk Manager"]
            badge_html = "".join(
                f"<span class='agent-badge'>{b}</span>" for b in badges
            )
            st.markdown(
                f"<div style='text-align:center;'>{badge_html}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #94a3b8; font-size: 0.75rem;'>"
        "⚠️ Disclaimer: This analysis is for educational purposes only. "
        "Investing in stock markets involves risk. "
        "Please consult a financial advisor before investing."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #64748b;'>"
        "Powered by Anthropic Claude 3 | Built with Streamlit | © 2026 Lakshya AI"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
