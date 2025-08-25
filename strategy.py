# strategy.py — Step 2 (VERBOSE VERSION)
# ------------------------------------------------------------
# Funksjonelt lik den kompakte Steg 2-strategien, men med rikelige
# kommentarer og hjelpefunksjoner for klarhet.
# ------------------------------------------------------------

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from indicators import rsi as _rsi, adx as _adx, atr as _atr


class TrendBias:
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Regime:
    RANGE = "range"
    TREND = "trend"
    STRONG_TREND = "strong_trend"


def ema(series, period):
    return pd.Series(series).ewm(span=int(period), adjust=False).mean().values


def _ema_series(series: pd.Series, period: int) -> pd.Series:
    return pd.Series(series).ewm(span=int(period), adjust=False).mean()


def _bollinger_bands_ewm(close: pd.Series, period: int, std_mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.ewm(span=int(period), adjust=False).mean()
    st = close.rolling(int(period)).std().fillna(method='bfill')
    upper = mid + float(std_mult) * st
    lower = mid - float(std_mult) * st
    return mid, upper, lower


def _safe_last(series: pd.Series | np.ndarray | list) -> float:
    if series is None:
        return 0.0
    try:
        if isinstance(series, (pd.Series, pd.Index)):
            return float(series.iloc[-1])
        if isinstance(series, (list, np.ndarray)):
            return float(series[-1])
    except Exception:
        pass
    return float(series)


def _nz(x: float, fallback: float = 0.0) -> float:
    try:
        if x is None:
            return fallback
        if isinstance(x, (float, int)):
            if math.isnan(x):
                return fallback
            return float(x)
        v = float(x)
        if math.isnan(v):
            return fallback
        return v
    except Exception:
        return fallback


_LAST_BAR_ID = {}
_LAST_BAR_ID_MICRO = {}

def should_emit(symbol: str, df_closed: pd.DataFrame, cooldown_bars: int = 2, throttle_seconds: int = 60) -> bool:
    global _LAST_BAR_ID
    try:
        bid = int(df_closed["close_time"].iloc[-1])
    except Exception:
        return True
    last = _LAST_BAR_ID.get(symbol)
    if last == bid:
        return False
    _LAST_BAR_ID[symbol] = bid
    return True


def mark_emitted(symbol: str, df_closed: pd.DataFrame):
    return


def should_emit_micro(symbol: str, df_closed: pd.DataFrame, cooldown_bars: int = 2, throttle_seconds: int = 60) -> bool:
    global _LAST_BAR_ID_MICRO
    try:
        bid = int(df_closed["close_time"].iloc[-1])
    except Exception:
        return True
    key = f"{symbol}|15m"
    last = _LAST_BAR_ID_MICRO.get(key)
    if last == bid:
        return False
    _LAST_BAR_ID_MICRO[key] = bid
    return True


def mark_emitted_micro(symbol: str, df_closed: pd.DataFrame):
    return


def _ema200_filter(df4h: pd.DataFrame, ema_period: int) -> tuple[float, float]:
    close = pd.Series(df4h["close"])
    ema200 = close.ewm(span=int(ema_period), adjust=False).mean()
    return float(close.iloc[-1]), float(ema200.iloc[-1])


def compute_trend_bias(df4h: pd.DataFrame, cfg: dict) -> str:
    ema_period = int(cfg.get("trend_filter", {}).get("ema_period", 200))
    adx_period = int(cfg.get("trend_filter", {}).get("adx_period", 14))
    adx_min = float(cfg.get("trend_filter", {}).get("adx_min", 20))

    last, ema_last = _ema200_filter(df4h, ema_period)
    adxv = _safe_last(_adx(df4h["high"], df4h["low"], df4h["close"], period=adx_period))

    if last > ema_last and adxv >= adx_min:
        return TrendBias.LONG
    if last < ema_last and adxv >= adx_min:
        return TrendBias.SHORT
    return TrendBias.FLAT


def compute_market_score(df4h: pd.DataFrame, df1h_closed: pd.DataFrame) -> tuple[float, str]:
    adx4 = _safe_last(_adx(df4h["high"], df4h["low"], df4h["close"], 14))
    atr_abs = _safe_last(_atr(df1h_closed["high"], df1h_closed["low"], df1h_closed["close"], 14))
    price = _nz(float(df1h_closed["close"].iloc[-1]), 1.0)
    atr_pct = atr_abs / max(1e-9, price) * 100.0

    score = max(0.0, min(100.0, 1.5 * float(adx4) + 3.0 * float(atr_pct)))
    if score >= 70.0:
        regime = Regime.STRONG_TREND
    elif score >= 35.0:
        regime = Regime.TREND
    else:
        regime = Regime.RANGE
    return float(score), regime


def compute_confidence(df4h: pd.DataFrame, df1h_closed: pd.DataFrame, bias: str, pb: bool, bo: bool) -> float:
    adx4 = _safe_last(_adx(df4h["high"], df4h["low"], df4h["close"], 14))
    atr_abs = _safe_last(_atr(df1h_closed["high"], df1h_closed["low"], df1h_closed["close"], 14))
    price = _nz(float(df1h_closed["close"].iloc[-1]), 1.0)
    atr_pct = atr_abs / max(1e-9, price)

    base = 0.5 * min(1.0, float(adx4) / 50.0) + 0.5 * min(1.0, float(atr_pct) * 8.0)
    if pb: base += 0.05
    if bo: base += 0.05
    return max(0.0, min(1.0, float(base)))


def _rolling_extreme(df: pd.DataFrame, lookback: int, bias: str) -> float:
    lb = max(2, int(lookback))
    if bias == TrendBias.LONG:
        return float(df["close"].rolling(lb).max().iloc[-2])
    else:
        return float(df["close"].rolling(lb).min().iloc[-2])


def _body_vs_atr_pct(df1h_closed: pd.DataFrame, atr_period: int) -> tuple[float, float]:
    body = abs(float(df1h_closed["close"].iloc[-1]) - float(df1h_closed["open"].iloc[-1]))
    atr_abs = _safe_last(_atr(df1h_closed["high"], df1h_closed["low"], df1h_closed["close"], atr_period))
    if atr_abs <= 0.0:
        return 0.0, 0.0
    return body, (body / atr_abs) * 100.0


def _atr_pct_now(df1h_closed: pd.DataFrame, atr_period: int) -> float:
    atr_abs = _safe_last(_atr(df1h_closed["high"], df1h_closed["low"], df1h_closed["close"], atr_period))
    price = _nz(float(df1h_closed["close"].iloc[-1]), 1.0)
    return float(atr_abs / max(1e-9, price) * 100.0)


def pullback_signal(df1h_closed: pd.DataFrame, bias: str, cfg: dict) -> bool:
    ema_fast = int(cfg.get("entries", {}).get("pullback", {}).get("ema_fast", 20))
    rsi_period = int(cfg.get("entries", {}).get("pullback", {}).get("rsi_period", 14))
    rsi_long_min = float(cfg.get("entries", {}).get("pullback", {}).get("rsi_long_min", 52))
    rsi_short_max = float(cfg.get("entries", {}).get("pullback", {}).get("rsi_short_max", 48))

    emaf = _ema_series(df1h_closed["close"], ema_fast)
    rsi = _rsi(df1h_closed["close"], period=rsi_period)

    if bias == TrendBias.LONG:
        return float(rsi.iloc[-1]) > rsi_long_min and float(df1h_closed["close"].iloc[-1]) >= float(emaf.iloc[-1])
    if bias == TrendBias.SHORT:
        return float(rsi.iloc[-1]) < rsi_short_max and float(df1h_closed["close"].iloc[-1]) <= float(emaf.iloc[-1])
    return False


def breakout_signal(df1h_closed: pd.DataFrame, bias: str, cfg: dict) -> bool:
    b = cfg.get("entries", {}).get("breakout", {})
    lookback = int(b.get("lookback", 10))
    vol_boost_pct = float(b.get("vol_boost_pct", 120.0))
    min_atr_pct = float(b.get("min_atr_pct", 0.25))
    atrp = int(cfg.get("stops", {}).get("atr_period", 14))

    level = _rolling_extreme(df1h_closed, lookback, bias)
    last_close = float(df1h_closed["close"].iloc[-1])
    base = (last_close > level) if bias == TrendBias.LONG else (last_close < level)
    if not base:
        return False

    _, body_pct = _body_vs_atr_pct(df1h_closed, atrp)
    atr_now_pct = _atr_pct_now(df1h_closed, atrp)
    return (body_pct >= vol_boost_pct) or (atr_now_pct >= min_atr_pct)


def mean_reversion_signal(df1h_closed: pd.DataFrame, cfg: dict) -> str | None:
    mr = cfg.get("entries", {}).get("mean_reversion", {})
    if not bool(mr.get("enabled", True)):
        return None
    rsi_period = int(mr.get("rsi_period", 14))
    rsi_low = float(mr.get("rsi_low", 35))
    rsi_high = float(mr.get("rsi_high", 65))
    bb_period = int(mr.get("bb_period", 20))
    bb_std = float(mr.get("bb_std", 2.0))

    close = pd.Series(df1h_closed["close"]).astype(float)
    rsi_v = _rsi(close, period=rsi_period)
    mid, upper, lower = _bollinger_bands_ewm(close, bb_period, bb_std)

    c = float(close.iloc[-1])
    if c <= float(lower.iloc[-1]) and float(rsi_v.iloc[-1]) <= rsi_low:
        return "BUY"
    if c >= float(upper.iloc[-1]) and float(rsi_v.iloc[-1]) >= rsi_high:
        return "SELL"
    return None


def _compute_atr(df1h_closed: pd.DataFrame, cfg: dict) -> float:
    period = int(cfg.get("stops", {}).get("atr_period", 14))
    return _nz(_safe_last(_atr(df1h_closed["high"], df1h_closed["low"], df1h_closed["close"], period)), 0.0)


def _r_distance(entry: float, stop: float) -> float:
    return abs(float(entry) - float(stop))


def _tp_partial_from_r(entry: float, rdist: float, r_multiple: float, side_long: bool) -> float:
    if side_long:
        return float(entry + r_multiple * rdist)
    else:
        return float(entry - r_multiple * rdist)


def build_order_plan(df1h_closed: pd.DataFrame, bias: str, cfg: dict) -> dict | None:
    a = _compute_atr(df1h_closed, cfg)
    if a <= 0.0:
        return None

    r_mult = float(cfg.get("stops", {}).get("r_mult", 1.0))
    trail_mult = float(cfg.get("trailing", {}).get("atr_mult", 2.3))
    entry = float(df1h_closed["close"].iloc[-1])

    if bias == TrendBias.LONG:
        stop = float(entry - r_mult * a)
        rdist = _r_distance(entry, stop)
        tp1 = _tp_partial_from_r(entry, rdist, 1.8, side_long=True)
    else:
        stop = float(entry + r_mult * a)
        rdist = _r_distance(entry, stop)
        tp1 = _tp_partial_from_r(entry, rdist, 1.8, side_long=False)

    return {
        "atr": float(a),
        "entry": float(entry),
        "stop": float(stop),
        "tp_partial": float(tp1),
        "r_distance": float(rdist),
        "trail_atr_mult": float(trail_mult),
    }


def get_atr_price_threshold(df1h_closed: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    a = _compute_atr(df1h_closed, cfg)
    p = _nz(float(df1h_closed["close"].iloc[-1]), 1.0)
    return float(a), float(a / max(1e-9, p))


def volatility_guard(df1h_closed: pd.DataFrame, regime: str, cfg: dict) -> tuple[bool, str, float, float]:
    atr_abs = _compute_atr(df1h_closed, cfg)
    price = _nz(float(df1h_closed["close"].iloc[-1]), 1.0)
    atr_pct = float(atr_abs / max(1e-9, price))
    limit = float(cfg.get("volatility_guard", {}).get("atr_price_cap", 0.12))
    if atr_pct >= limit:
        return True, f"ATR/price {atr_pct:.2%} >= cap {limit:.2%}", atr_pct, limit
    return False, "-", atr_pct, limit


def micro_signal_ema_stoch(df15_closed: pd.DataFrame, bias_4h: str, cfg: dict) -> str | None:
    mcfg = cfg.get("micro_entry", {})
    ema_fast = int(mcfg.get("ema_fast", 21))
    ema_slow = int(mcfg.get("ema_slow", 55))
    rsi_period = int(mcfg.get("rsi_period", 14))
    long_min = float(mcfg.get("rsi_long_min", 55))
    short_max = float(mcfg.get("rsi_short_max", 45))

    close = pd.Series(df15_closed["close"]).astype(float)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    rsi_v = _rsi(close, period=rsi_period)

    if bias_4h == TrendBias.LONG:
        if float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]) and float(rsi_v.iloc[-1]) >= long_min:
            return "long"
        return None

    if bias_4h == TrendBias.SHORT:
        if float(ema_f.iloc[-1]) < float(ema_s.iloc[-1]) and float(rsi_v.iloc[-1]) <= short_max:
            return "short"
        return None

    return None

# ==== Tillegg: ATR-median og killswitch-state for 15m ====
def _atr_median(df_closed: pd.DataFrame, period: int = 14, lookback: int = 200) -> tuple[float, float]:
    try:
        if df_closed is None or len(df_closed) < max(period+2, 20):
            return 0.0, 0.0
        a = _atr(df_closed["high"], df_closed["low"], df_closed["close"], period=period)
        a = pd.Series(a).astype(float)
        cur = float(a.iloc[-1])
        med = float(pd.Series(a.iloc[-int(lookback):]).median())
        if med != med:  # NaN
            med = 0.0
        if cur != cur:
            cur = 0.0
        return cur, med
    except Exception:
        return 0.0, 0.0

# Global: når (epoch sek.) et symbol er lov å trade igjen etter ATR-spike
KILL_UNTIL = {}
