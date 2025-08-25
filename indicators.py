# indicators.py — Step 2: Wilder RSI/ADX (backwards compatible)

import numpy as np
import pandas as pd

def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    # Wilder smoothing == EMA with alpha = 1/period
    return series.ewm(alpha=1/period, adjust=False).mean()

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.Series(close).astype(float)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = _wilder_ema(up, period)
    avg_loss = _wilder_ema(down, period)
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi

def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0.0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0.0), 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = _wilder_ema(tr, period)
    pdi = 100 * _wilder_ema(plus_dm, period) / atr.replace(0, np.nan)
    ndi = 100 * _wilder_ema(minus_dm, period) / atr.replace(0, np.nan)

    dx = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)) * 100.0
    adx = _wilder_ema(dx, period).fillna(0.0)
    return adx

# Backwards compatible names
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return rsi_wilder(close, period)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return adx_wilder(high, low, close, period)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)
    tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return _wilder_ema(tr, period).bfill()
