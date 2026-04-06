"""
╔══════════════════════════════════════════════════════════════╗
║           ULTIMATE SWING SCANNER BOT v2.0                   ║
║  Strategy: MACD (D/W/M) + 4H EMA Stack + Stoch + Filters   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID         = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")
SCAN_INTERVAL   = 4 * 60 * 60   # every 4 hours
MAX_RESULTS     = 15             # top N per direction to send

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── MARKET HOURS CHECK ────────────────────────────────────────────────────────
def is_market_open():
    """Only scan during/around US market hours (don't waste at 3am)."""
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    # Mon-Fri only, between 6am and 8pm ET (pre/post market included)
    if now.weekday() >= 5:
        return False
    return 6 <= now.hour <= 20

# ── STOCK LIST ────────────────────────────────────────────────────────────────
def get_stock_list():
    tickers = set()

    # S&P 500
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        for t in sp500["Symbol"].tolist():
            tickers.add(t.replace(".", "-"))
    except Exception as e:
        logger.warning(f"SP500 fetch failed: {e}")

    # NASDAQ 100
    try:
        ndx = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
        col = [c for c in ndx.columns if "ticker" in c.lower() or "symbol" in c.lower()]
        if col:
            for t in ndx[col[0]].tolist():
                tickers.add(str(t).replace(".", "-"))
    except Exception as e:
        logger.warning(f"NDX fetch failed: {e}")

    # Russell 1000 high-volume popular names + sectors
    extra = [
        # Mega cap tech
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AMD","NFLX",
        "INTC","CSCO","ADBE","QCOM","TXN","AVGO","MU","LRCX","KLAC","AMAT",
        # Cloud/SaaS
        "CRM","NOW","SNOW","PLTR","CRWD","ZS","DDOG","NET","MDB","BILL","HUBS",
        "OKTA","TWLO","DOCN","GTLB","CFLT","S","ESTC","APPN","BRZE",
        # Fintech
        "COIN","HOOD","SOFI","SQ","PYPL","V","MA","AXP","GS","JPM","BAC","C",
        "WFC","MS","BLK","SCHW","IBKR","NU","AFRM","UPST",
        # EV / Clean energy
        "RIVN","LCID","NIO","XPEV","LI","CHPT","BLNK","EVGO","TSLA","PLUG","FCEL",
        # China ADR
        "BIDU","JD","BABA","PDD","MELI","SE","GRAB",
        # Consumer / retail
        "ABNB","UBER","LYFT","DASH","SNAP","PINS","SPOT","RBLX","U","ETSY",
        "SHOP","WMT","TGT","COST","HD","LOW","NKE","LULU","DKS",
        # Healthcare / Biotech
        "LLY","NVO","MRNA","BNTX","ABBV","JNJ","PFE","AMGN","GILD","REGN",
        "VRTX","BIIB","ILMN","TDOC","HIMS",
        # Energy
        "XOM","CVX","COP","SLB","HAL","OXY","MPC","VLO","PSX","DVN",
        # AI / Emerging tech
        "AI","BBAI","SOUN","IONQ","RGTI","QUBT","QBTS","ACHR","JOBY","RKLB",
        "LUNR","ASTS","IREN","CORZ","HUT","MARA","RIOT","CLSK",
        # ETFs (useful benchmarks)
        "SPY","QQQ","IWM","SMH","XLK","XLF","XLE","XLV","ARKK","SOXL",
    ]
    tickers.update(extra)
    result = sorted(tickers)
    logger.info(f"Total tickers to scan: {len(result)}")
    return result


# ── EARNINGS CHECK ────────────────────────────────────────────────────────────
def days_to_earnings(ticker: str) -> int | None:
    """Return days until next earnings. None if unknown."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is not None and not cal.empty:
            # calendar is a DataFrame with dates as columns
            dates = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else None
            if dates is not None:
                for d in dates:
                    if pd.notna(d):
                        days = (pd.Timestamp(d).date() - datetime.now().date()).days
                        if days >= 0:
                            return days
    except Exception:
        pass
    return None


# ── INDICATORS ────────────────────────────────────────────────────────────────
def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def compute_macd(close):
    m = ema(close, 12) - ema(close, 26)
    sig = ema(m, 9)
    return m, sig, m - sig

def compute_stoch(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    pct_k = 100 * (close - ll) / (hh - ll + 1e-10)
    slow_k = pct_k.rolling(3).mean()
    slow_d = slow_k.rolling(d).mean()
    return slow_k, slow_d

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_volume_ratio(volume, period=20):
    """Current volume vs average — >1.5 means above average."""
    avg = volume.rolling(period).mean()
    return volume.iloc[-1] / (avg.iloc[-1] + 1e-10)

def compute_emas(close):
    return {k: ema(close, v) for k, v in
            [("e9",9),("e21",21),("e50",50),("e100",100),("e200",200)]}

def relative_strength(close, spy_close):
    """RS ratio vs SPY over last 50 bars."""
    if len(close) < 50 or len(spy_close) < 50:
        return 1.0
    stock_ret = close.iloc[-1] / close.iloc[-50]
    spy_ret   = spy_close.iloc[-1] / spy_close.iloc[-50]
    return round(stock_ret / (spy_ret + 1e-10), 2)


# ── SIGNAL CHECKS ─────────────────────────────────────────────────────────────
def macd_bull(macd, sig, hist):
    if len(hist) < 3: return False
    return (macd.iloc[-1] > sig.iloc[-1] and
            hist.iloc[-1] > 0 and
            hist.iloc[-1] > hist.iloc[-2])

def macd_bear(macd, sig, hist):
    if len(hist) < 3: return False
    return (macd.iloc[-1] < sig.iloc[-1] and
            hist.iloc[-1] < 0 and
            hist.iloc[-1] < hist.iloc[-2])

def stoch_bull_entry(k, d):
    """Fresh crossup from oversold — best entry zone."""
    if len(k) < 4: return False, 0
    cross_up   = k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]
    from_below = k.iloc[-3] < 40
    not_hot    = k.iloc[-1] < 72
    score = sum([cross_up, from_below, not_hot, k.iloc[-1] < 55])
    return cross_up and from_below and not_hot, round(k.iloc[-1], 1)

def stoch_bear_entry(k, d):
    """Fresh crossdown from overbought — best short entry."""
    if len(k) < 4: return False, 0
    cross_dn   = k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2]
    from_above = k.iloc[-3] > 60
    not_cold   = k.iloc[-1] > 28
    score = sum([cross_dn, from_above, not_cold, k.iloc[-1] > 45])
    return cross_dn and from_above and not_cold, round(k.iloc[-1], 1)

def ema_bull_stack(close, emas):
    p   = close.iloc[-1]
    e9  = emas["e9"].iloc[-1]
    e21 = emas["e21"].iloc[-1]
    e50 = emas["e50"].iloc[-1]
    e200= emas["e200"].iloc[-1]
    stack  = p > e9 > e21 > e50
    above  = p > e200
    slope9 = emas["e9"].iloc[-1]  > emas["e9"].iloc[-4]
    slope21= emas["e21"].iloc[-1] > emas["e21"].iloc[-4]
    return stack and above and slope9 and slope21

def ema_bear_stack(close, emas):
    p   = close.iloc[-1]
    e9  = emas["e9"].iloc[-1]
    e21 = emas["e21"].iloc[-1]
    e50 = emas["e50"].iloc[-1]
    e200= emas["e200"].iloc[-1]
    stack  = p < e9 < e21 < e50
    below  = p < e200
    slope9 = emas["e9"].iloc[-1]  < emas["e9"].iloc[-4]
    slope21= emas["e21"].iloc[-1] < emas["e21"].iloc[-4]
    return stack and below and slope9 and slope21

def not_overextended(close, emas, threshold=7.0):
    """Price must be within threshold% of EMA21 — no chasing pumps."""
    p   = close.iloc[-1]
    e21 = emas["e21"].iloc[-1]
    return abs(p - e21) / e21 * 100 < threshold

def near_ema_support(close, emas):
    """Price pulling back to EMA21 or EMA50 — ideal entry zone."""
    p   = close.iloc[-1]
    e21 = emas["e21"].iloc[-1]
    e50 = emas["e50"].iloc[-1]
    near21 = abs(p - e21) / e21 * 100 < 2.5
    near50 = abs(p - e50) / e50 * 100 < 2.5
    return near21, near50

def compute_sl_tp(close, high, low, emas, direction="bull", atr=None):
    """Calculate suggested stop loss and take profit levels."""
    price = close.iloc[-1]
    e50   = emas["e50"].iloc[-1]
    e21   = emas["e21"].iloc[-1]

    # Use ATR for dynamic stops
    atr_val = atr.iloc[-1] if atr is not None else price * 0.02

    if direction == "bull":
        # Stop: just below EMA50 or 1.5x ATR below entry
        sl = round(min(e50 * 0.995, price - 1.5 * atr_val), 2)
        # TP1: 1.5x R, TP2: 3x R
        risk = price - sl
        tp1  = round(price + 1.5 * risk, 2)
        tp2  = round(price + 3.0 * risk, 2)
        rr   = round((tp1 - price) / (price - sl), 2) if price > sl else 0
    else:
        sl   = round(max(e50 * 1.005, price + 1.5 * atr_val), 2)
        risk = sl - price
        tp1  = round(price - 1.5 * risk, 2)
        tp2  = round(price - 3.0 * risk, 2)
        rr   = round((price - tp1) / (sl - price), 2) if sl > price else 0

    return sl, tp1, tp2, rr

def detect_divergence(close, macd, lookback=10):
    """Detect bullish divergence: price lower low, MACD higher low."""
    if len(close) < lookback + 2:
        return False
    price_ll = close.iloc[-1] < close.iloc[-lookback:-1].min()
    macd_hl  = macd.iloc[-1] > macd.iloc[-lookback:-1].min()
    return price_ll and macd_hl

def detect_bear_divergence(close, macd, lookback=10):
    """Detect bearish divergence: price higher high, MACD lower high."""
    if len(close) < lookback + 2:
        return False
    price_hh = close.iloc[-1] > close.iloc[-lookback:-1].max()
    macd_lh  = macd.iloc[-1] < macd.iloc[-lookback:-1].max()
    return price_hh and macd_lh


# ── TIMEFRAME ANALYSIS ────────────────────────────────────────────────────────
def analyze_tf(df):
    if df is None or len(df) < 60:
        return None
    try:
        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()
        vol   = df["Volume"].dropna() if "Volume" in df.columns else None

        if len(close) < 55:
            return None

        m, sig, hist = compute_macd(close)
        k, d         = compute_stoch(high, low, close)
        emas_        = compute_emas(close)
        rsi_         = compute_rsi(close)
        atr_         = compute_atr(high, low, close)
        vol_ratio    = compute_volume_ratio(vol) if vol is not None else 1.0

        return {
            "macd_bull":    macd_bull(m, sig, hist),
            "macd_bear":    macd_bear(m, sig, hist),
            "macd_line":    round(m.iloc[-1], 4),
            "macd_hist":    round(hist.iloc[-1], 4),
            "stoch_k":      round(k.iloc[-1], 1),
            "ema_bull":     ema_bull_stack(close, emas_),
            "ema_bear":     ema_bear_stack(close, emas_),
            "not_extended": not_overextended(close, emas_),
            "near_e21":     near_ema_support(close, emas_)[0],
            "near_e50":     near_ema_support(close, emas_)[1],
            "rsi":          round(rsi_.iloc[-1], 1),
            "atr":          atr_,
            "emas":         emas_,
            "close":        close,
            "vol_ratio":    round(vol_ratio, 2),
            "divergence_bull": detect_divergence(close, m),
            "divergence_bear": detect_bear_divergence(close, m),
        }
    except Exception as e:
        logger.debug(f"analyze_tf error: {e}")
        return None


# ── FETCH ALL TIMEFRAMES ──────────────────────────────────────────────────────
def fetch_all(ticker: str, spy_close=None):
    try:
        t = yf.Ticker(ticker)
        data = {}

        # 4H via 1H resample
        h1 = t.history(period="60d", interval="1h")
        if h1 is not None and len(h1) > 30:
            h4 = h1.resample("4h").agg({
                "Open":"first","High":"max","Low":"min",
                "Close":"last","Volume":"sum"
            }).dropna()
            data["4h"] = h4

        # Daily, Weekly, Monthly
        d1 = t.history(period="2y",  interval="1d")
        wk = t.history(period="5y",  interval="1wk")
        mo = t.history(period="10y", interval="1mo")

        if d1 is not None and len(d1) > 60:  data["1d"]  = d1
        if wk is not None and len(wk) > 50:  data["1wk"] = wk
        if mo is not None and len(mo) > 24:  data["1mo"] = mo

        if len(data) < 4:
            return None

        # RS vs SPY
        rs = 1.0
        if spy_close is not None and "1d" in data:
            rs = relative_strength(data["1d"]["Close"], spy_close)

        data["rs"] = rs
        return data
    except Exception as e:
        logger.debug(f"fetch_all {ticker}: {e}")
        return None


# ── FULL SCORE ────────────────────────────────────────────────────────────────
def score_ticker(ticker: str, spy_close=None):
    data = fetch_all(ticker, spy_close)
    if not data:
        return None

    tf4  = analyze_tf(data.get("4h"))
    tf1d = analyze_tf(data.get("1d"))
    tf1w = analyze_tf(data.get("1wk"))
    tf1m = analyze_tf(data.get("1mo"))
    rs   = data.get("rs", 1.0)

    if not all([tf4, tf1d, tf1w, tf1m]):
        return None

    price = round(tf4["close"].iloc[-1], 2)

    # ── STRONG BUY ────────────────────────────────────────────────────────────
    macd_3tf_bull = tf1d["macd_bull"] and tf1w["macd_bull"] and tf1m["macd_bull"]
    stoch_ok, stoch_k = stoch_bull_entry(
        pd.Series([tf4["stoch_k"]]),
        pd.Series([tf4["stoch_k"]])
    )
    # Re-run stoch properly on 4H df
    h4df = data.get("4h")
    if h4df is not None and len(h4df) > 20:
        k4, d4 = compute_stoch(h4df["High"], h4df["Low"], h4df["Close"])
        stoch_ok, stoch_k_val = stoch_bull_entry(k4, d4)
    else:
        stoch_ok, stoch_k_val = False, 0

    if (macd_3tf_bull and tf4["ema_bull"] and stoch_ok and tf4["not_extended"]):

        # Calculate SL/TP
        sl, tp1, tp2, rr = compute_sl_tp(
            tf4["close"], h4df["High"], h4df["Low"],
            tf4["emas"], "bull", tf4["atr"]
        )

        # Only take if R:R >= 1.5
        if rr < 1.5:
            return None

        # Earnings warning
        earn_days = days_to_earnings(ticker)

        # Score out of 10
        score = sum([
            macd_3tf_bull * 3,           # 3 pts — core filter
            tf4["ema_bull"] * 2,         # 2 pts — 4H structure
            stoch_ok,                    # 1 pt  — entry timing
            tf4["near_e21"] or tf4["near_e50"],  # 1 pt — pullback quality
            tf4["vol_ratio"] > 1.3,      # 1 pt  — volume confirms
            tf1d["divergence_bull"],     # 1 pt  — divergence bonus
            rs > 1.1,                    # 1 pt  — outperforming SPY
        ])

        return {
            "signal":     "BUY",
            "ticker":     ticker,
            "price":      price,
            "score":      score,
            "sl":         sl,
            "tp1":        tp1,
            "tp2":        tp2,
            "rr":         rr,
            "rsi":        tf4["rsi"],
            "stoch_k":    stoch_k_val,
            "vol_ratio":  tf4["vol_ratio"],
            "rs":         rs,
            "divergence": tf1d["divergence_bull"],
            "near_ema":   "EMA21" if tf4["near_e21"] else ("EMA50" if tf4["near_e50"] else "—"),
            "earn_days":  earn_days,
            "macd_4h":    tf4["macd_bull"],
        }

    # ── STRONG SHORT ──────────────────────────────────────────────────────────
    macd_3tf_bear = tf1d["macd_bear"] and tf1w["macd_bear"] and tf1m["macd_bear"]

    if h4df is not None and len(h4df) > 20:
        k4, d4 = compute_stoch(h4df["High"], h4df["Low"], h4df["Close"])
        stoch_short_ok, stoch_k_val = stoch_bear_entry(k4, d4)
    else:
        stoch_short_ok, stoch_k_val = False, 0

    if (macd_3tf_bear and tf4["ema_bear"] and stoch_short_ok and tf4["not_extended"]):

        sl, tp1, tp2, rr = compute_sl_tp(
            tf4["close"], h4df["High"], h4df["Low"],
            tf4["emas"], "bear", tf4["atr"]
        )

        if rr < 1.5:
            return None

        earn_days = days_to_earnings(ticker)

        score = sum([
            macd_3tf_bear * 3,
            tf4["ema_bear"] * 2,
            stoch_short_ok,
            tf4["near_e21"] or tf4["near_e50"],
            tf4["vol_ratio"] > 1.3,
            tf1d["divergence_bear"],
            rs < 0.9,
        ])

        return {
            "signal":     "SHORT",
            "ticker":     ticker,
            "price":      price,
            "score":      score,
            "sl":         sl,
            "tp1":        tp1,
            "tp2":        tp2,
            "rr":         rr,
            "rsi":        tf4["rsi"],
            "stoch_k":    stoch_k_val,
            "vol_ratio":  tf4["vol_ratio"],
            "rs":         rs,
            "divergence": tf1d["divergence_bear"],
            "near_ema":   "EMA21" if tf4["near_e21"] else ("EMA50" if tf4["near_e50"] else "—"),
            "earn_days":  earn_days,
            "macd_4h":    tf4["macd_bear"],
        }

    return None


# ── FORMAT ALERT ──────────────────────────────────────────────────────────────
def format_alert(r: dict) -> str:
    is_buy    = r["signal"] == "BUY"
    emoji     = "🟢" if is_buy else "🔴"
    direction = "LONG 📈" if is_buy else "SHORT 📉"
    stars     = "⭐" * min(r["score"], 10)

    # Earnings warning
    earn_str = ""
    if r["earn_days"] is not None:
        if r["earn_days"] <= 7:
            earn_str = f"\n⚠️ *EARNINGS IN {r['earn_days']} DAYS — CAUTION*"
        elif r["earn_days"] <= 14:
            earn_str = f"\n📅 Earnings in {r['earn_days']} days"

    # Volume
    vol_str = f"🔥 HIGH" if r["vol_ratio"] > 1.5 else (
              f"✅ AVG+" if r["vol_ratio"] > 1.1 else "🔵 Normal")

    # Divergence
    div_str = "✅ YES" if r["divergence"] else "—"

    # RS
    rs_str = f"{'🚀' if r['rs'] > 1.2 else '✅'} {r['rs']}x SPY" if r["rs"] > 1.0 else f"⚠️ {r['rs']}x SPY"

    # 4H MACD bonus
    macd4h_str = "✅ Confirmed" if r["macd_4h"] else "⏳ Pending"

    msg = (
        f"{emoji} *{r['ticker']}* — {direction}\n"
        f"💰 Price: *${r['price']}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *ENTRY:* ${r['price']}\n"
        f"🛑 *STOP LOSS:* ${r['sl']}\n"
        f"💵 *TP1:* ${r['tp1']}\n"
        f"🏆 *TP2:* ${r['tp2']}\n"
        f"📐 *R:R Ratio:* 1:{r['rr']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 *INDICATORS*\n"
        f"• RSI (4H): {r['rsi']}\n"
        f"• Stoch %K: {r['stoch_k']}\n"
        f"• Volume:  {vol_str} ({r['vol_ratio']}x avg)\n"
        f"• Near EMA: {r['near_ema']}\n"
        f"• 4H MACD: {macd4h_str}\n"
        f"• Divergence: {div_str}\n"
        f"• RS vs SPY: {rs_str}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⚡ *Confidence:* {stars} ({r['score']}/10)"
        f"{earn_str}\n"
    )
    return msg


# ── MAIN SCAN ─────────────────────────────────────────────────────────────────
async def run_scan(bot: Bot, manual=False):
    now_et = datetime.now(pytz.timezone("America/New_York"))
    logger.info(f"Scan started at {now_et.strftime('%Y-%m-%d %H:%M ET')}")

    # Pre-fetch SPY for relative strength
    spy_close = None
    try:
        spy_data  = yf.Ticker("SPY").history(period="1y", interval="1d")
        spy_close = spy_data["Close"].dropna()
    except Exception:
        pass

    tickers = get_stock_list()
    buys, shorts = [], []
    errors = 0

    # Send progress update
    await bot.send_message(
        chat_id=CHAT_ID,
        text=f"🔍 Scan started — checking {len(tickers)} stocks...\n"
             f"🕐 {now_et.strftime('%H:%M ET')} | Results in ~15 mins"
    )

    for i, ticker in enumerate(tickers):
        try:
            result = score_ticker(ticker, spy_close)
            if result:
                if result["signal"] == "BUY":
                    buys.append(result)
                    logger.info(f"✅ BUY: {ticker} score={result['score']} rr={result['rr']}")
                else:
                    shorts.append(result)
                    logger.info(f"🔻 SHORT: {ticker} score={result['score']} rr={result['rr']}")
        except Exception as e:
            errors += 1
            logger.debug(f"Error {ticker}: {e}")

        if i % 100 == 0 and i > 0:
            await asyncio.sleep(3)
            logger.info(f"Progress: {i}/{len(tickers)}")

    # Sort by score desc, then R:R desc
    buys.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    shorts.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)

    await send_full_report(bot, buys, shorts, len(tickers), errors)


# ── SEND REPORT ───────────────────────────────────────────────────────────────
async def send_full_report(bot, buys, shorts, total, errors):
    now = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M UTC")
    et  = datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M ET")

    # Summary header
    header = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *SWING SCANNER REPORT*\n"
        f"🕐 {now} ({et})\n"
        f"🔍 {total} stocks scanned\n"
        f"🟢 {len(buys)} Buys | 🔴 {len(shorts)} Shorts\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
    await bot.send_message(chat_id=CHAT_ID, text=header, parse_mode="Markdown")
    await asyncio.sleep(0.5)

    # Send each BUY individually (rich format)
    if buys:
        await bot.send_message(
            chat_id=CHAT_ID,
            text=f"🟢 *TOP LONG SETUPS* ({len(buys)} found)",
            parse_mode="Markdown"
        )
        for r in buys[:MAX_RESULTS]:
            try:
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=format_alert(r),
                    parse_mode="Markdown"
                )
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"Failed to send alert for {r['ticker']}: {e}")
    else:
        await bot.send_message(
            chat_id=CHAT_ID,
            text="🟢 No strong buy setups this scan — market may be extended or choppy."
        )

    await asyncio.sleep(1)

    # Send each SHORT individually
    if shorts:
        await bot.send_message(
            chat_id=CHAT_ID,
            text=f"🔴 *TOP SHORT SETUPS* ({len(shorts)} found)",
            parse_mode="Markdown"
        )
        for r in shorts[:MAX_RESULTS]:
            try:
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=format_alert(r),
                    parse_mode="Markdown"
                )
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"Failed to send short alert for {r['ticker']}: {e}")
    else:
        await bot.send_message(
            chat_id=CHAT_ID,
            text="🔴 No strong short setups this scan."
        )

    # Footer
    footer = (
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "📋 *HOW TO USE THESE ALERTS*\n"
        "1. Entry = current price zone\n"
        "2. Set your stop loss at the SL level\n"
        "3. TP1 = take 50-60% off the table\n"
        "4. TP2 = let rest run with trailing stop\n"
        "5. ⚠️ SKIP any trade with earnings <7 days\n"
        "6. Always glance at the chart before entering\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ _Not financial advice. Always do your own analysis._"
    )
    await bot.send_message(chat_id=CHAT_ID, text=footer, parse_mode="Markdown")
    logger.info(f"Report sent. Buys: {len(buys)}, Shorts: {len(shorts)}, Errors: {errors}")


# ── BOT COMMANDS ──────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"🚀 *ULTIMATE SWING SCANNER BOT*\n\n"
        f"Your Chat ID: `{cid}`\n\n"
        f"*Commands:*\n"
        f"/scan — Manual scan now (~15 mins)\n"
        f"/status — Bot status\n"
        f"/help — How to use alerts\n\n"
        f"✅ Auto scan every 4 hours\n"
        f"📊 Strategy: MACD(D/W/M) + 4H EMA + Stoch\n"
        f"🎯 Includes: Entry, SL, TP1, TP2, R:R\n"
        f"📅 Earnings warnings included\n"
        f"📈 Volume & RS vs SPY filters",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    et = datetime.now(pytz.timezone("America/New_York"))
    market = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
    await update.message.reply_text(
        f"✅ *Bot is running*\n\n"
        f"📡 Scans ~2000 stocks every 4 hours\n"
        f"🕐 NY Time: {et.strftime('%H:%M ET')}\n"
        f"🏛 Market: {market}\n\n"
        f"*Strategy filters:*\n"
        f"• MACD bullish on Daily + Weekly + Monthly\n"
        f"• 4H EMA stack aligned\n"
        f"• Stochastic fresh crossover entry\n"
        f"• Not overextended (no chasing pumps)\n"
        f"• R:R minimum 1.5:1\n"
        f"• Volume confirmation\n"
        f"• Earnings date warning",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📋 *HOW TO USE YOUR ALERTS*\n\n"
        "*ENTRY ZONE*\n"
        "Enter at or near the price shown. Don't chase if it moves more than 1% away.\n\n"
        "*STOP LOSS*\n"
        "Set your SL immediately when you enter. Non-negotiable. This is below EMA50.\n\n"
        "*TP1 (Take Profit 1)*\n"
        "Close 50-60% of your position here. Lock in profit.\n\n"
        "*TP2 (Take Profit 2)*\n"
        "Let the rest ride. Move your stop to breakeven after TP1 hits.\n\n"
        "*R:R RATIO*\n"
        "Only trades with 1:1.5+ ratio are sent. Higher = better.\n\n"
        "*⚠️ EARNINGS WARNING*\n"
        "If earnings are <7 days away — SKIP or reduce size significantly.\n\n"
        "*VOLUME*\n"
        "🔥 HIGH volume = strongest signal. Normal volume = still valid.\n\n"
        "*RS vs SPY*\n"
        "Stock outperforming S&P500 = stronger candidate.\n\n"
        "*DIVERGENCE*\n"
        "If shown = extra confirmation of reversal. High probability bonus.",
        parse_mode="Markdown"
    )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Manual scan started... check back in ~15 mins")
    await run_scan(context.bot, manual=True)

async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Scheduled scan triggered")
    await run_scan(context.bot)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise ValueError("❌ Set TELEGRAM_TOKEN environment variable!")
    if CHAT_ID == "YOUR_CHAT_ID_HERE":
        raise ValueError("❌ Set CHAT_ID environment variable!")

    logger.info("Starting Ultimate Swing Scanner Bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("scan",   cmd_scan))
    app.add_handler(CommandHandler("help",   cmd_help))

    app.job_queue.run_repeating(
        scheduled_scan,
        interval=SCAN_INTERVAL,
        first=90  # first scan 90 seconds after startup
    )

    logger.info("Bot polling started ✅")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
