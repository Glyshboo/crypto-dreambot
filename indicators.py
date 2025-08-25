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

# === Soft-filter indikatorer (trygge å importere) ===
def vwap_session_from_df(df):
    """
    Session-VWAP fra dagens start (UTC) basert på typisk pris ((H+L+C)/3) og vol.
    Forventer kolonner: 'high','low','close','volume'. Index: tidsstempel.
    """
    import pandas as _pd
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    for c in ("high","low","close","volume"):
        if c not in d.columns:
            # fallbacks
            if c == "volume" and "vol" in d.columns:
                d["volume"] = d["vol"]
            else:
                return None
    d = d.tail(500)  # nok til en dag
    d["_date"] = _pd.to_datetime(d.index, utc=True).date
    today = _pd.to_datetime(_pd.Timestamp.utcnow().date())
    # behandle bare dagens linjer (UTC)
    d = d[d["_date"] == today.to_pydatetime().date()]
    if d.empty:
        # fall-back: bruk siste 24 bar
        d = df.tail(24).copy()
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    pv = tp * d["volume"]
    cum_pv = pv.cumsum()
    cum_v  = d["volume"].cumsum().replace(0, float("nan"))
    vwap = (cum_pv / cum_v).iloc[-1]
    return float(vwap) if vwap == vwap else None  # nan-sjekk

def obv_from_df(df):
    """
    On-Balance Volume for df (kolonner: close, volume/vol).
    Returnerer en pandas.Series (OBV).
    """
    import pandas as _pd
    if df is None or len(df) == 0:
        return _pd.Series([], dtype=float)
    d = df.copy()
    if "volume" not in d.columns:
        if "vol" in d.columns:
            d["volume"] = d["vol"]
        else:
            d["volume"] = 0.0
    close = d["close"].values
    vol = d["volume"].values
    obv = [0.0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + vol[i])
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - vol[i])
        else:
            obv.append(obv[-1])
    s = _pd.Series(obv, index=d.index)
    return s

def supertrend_from_df(df, period=10, multiplier=3.0):
    """
    Returnerer (st_line, st_dir) der st_dir=+1 bull, -1 bear.
    Implementerer en enkel Supertrend basert på ATR.
    """
    import pandas as _pd
    import numpy as _np
    if df is None or len(df) < period+2:
        return None, 0
    d = df.copy()
    H, L, C = d["high"], d["low"], d["close"]
    # ATR
    tr = _pd.concat([
        (H - L).abs(),
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()

    # Basisbånd
    hl2 = (H + L) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    # Supertrend retning
    st = _pd.Series(index=d.index, dtype=float)
    dirn = _pd.Series(index=d.index, dtype=int)
    st.iloc[0] = hl2.iloc[0]
    dirn.iloc[0] = 1
    for i in range(1, len(d)):
        if C.iloc[i] > upper.iloc[i-1]:
            dirn.iloc[i] = 1
        elif C.iloc[i] < lower.iloc[i-1]:
            dirn.iloc[i] = -1
        else:
            dirn.iloc[i] = dirn.iloc[i-1]
            if dirn.iloc[i] == 1:
                upper.iloc[i] = min(upper.iloc[i], upper.iloc[i-1])
            else:
                lower.iloc[i] = max(lower.iloc[i], lower.iloc[i-1])
        st.iloc[i] = lower.iloc[i] if dirn.iloc[i] == 1 else upper.iloc[i]
    return float(st.iloc[-1]), int(dirn.iloc[-1])
