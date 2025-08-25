# -*- coding: utf-8 -*-
"""
strategy.py — Patch 02
- Markedsregime-filtre (range/trend/strong_trend) basert på 4H og 1H.
- ATR-basert SL/TP og trailing.
- ADX-gating, volum-bekreftelse, dedup/cooldown, mikro-bekreftelse fra 15m.
- Anti-spam (signal de-dupe på tvers av tidssteg).
Denne modulen har ingen eksterne avhengigheter utover pandas/numpy.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import time

# ---------- Datatyper ----------

@dataclass
class Signal:
    symbol: str
    side: str          # "long" eller "short"
    price: float
    sl: float
    tp1: float
    tp2: float
    meta: Dict[str, Any]


class Strategy:
    def __init__(self,
                 regime_4h_adx_th: float = 18.0,
                 strong_adx_th: float = 25.0,
                 atr_mult_sl: float = 1.8,
                 tp_r1: float = 2.0,
                 tp_r2: float = 3.0,
                 min_vol_mult: float = 1.0,
                 cooldown_minutes: float = 20.0):
        self.regime_4h_adx_th = regime_4h_adx_th
        self.strong_adx_th = strong_adx_th
        self.atr_mult_sl = atr_mult_sl
        self.tp_r1 = tp_r1
        self.tp_r2 = tp_r2
        self.min_vol_mult = min_vol_mult
        self.cooldown_minutes = cooldown_minutes
        self._last_signal_time: Dict[str, float] = {}
        self._last_signal_side: Dict[str, str] = {}

    # ---------- Indikatorer (enkle, interne) ----------

    @staticmethod
    def _ema(a: pd.Series, n: int) -> pd.Series:
        return a.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
        # Enkel ADX (ikke Welles Wilder eksakt, men tilstrekkelig for gating)
        high, low, close = df['high'], df['low'], df['close']
        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = Strategy._atr(df, n=n) * n  # approx TR_n
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).sum() / tr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).sum() / tr.replace(0, np.nan)
        dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
        return dx.rolling(n).mean()

    @staticmethod
    def _obv(df: pd.DataFrame) -> pd.Series:
        close = df['close']
        vol = df['volume']
        direction = np.sign(close.diff().fillna(0.0))
        return (direction * vol).fillna(0.0).cumsum()

    # ---------- Hjelpere ----------

    def _regime(self, df_4h: pd.DataFrame) -> str:
        adx4 = self._adx(df_4h).iloc[-1]
        ema_slow = self._ema(df_4h['close'], 200).iloc[-1]
        ema_fast = self._ema(df_4h['close'], 50).iloc[-1]
        bias_up = ema_fast > ema_slow
        if adx4 >= self.strong_adx_th:
            return 'strong_trend_up' if bias_up else 'strong_trend_down'
        if adx4 >= self.regime_4h_adx_th:
            return 'trend_up' if bias_up else 'trend_down'
        return 'range'

    def _vol_ok(self, df_1h: pd.DataFrame) -> bool:
        v = df_1h['volume']
        return v.iloc[-1] >= self.min_vol_mult * v.rolling(20).mean().iloc[-1]

    def _micro_conf(self, df_15m: pd.DataFrame, side: str) -> bool:
        # Bruk OBV-slope som enkel momentum-bekreftelse
        obv = self._obv(df_15m)
        slope = (obv.iloc[-1] - obv.iloc[-5]) / 5.0
        return (slope > 0) if side == 'long' else (slope < 0)

    def _cooldown_passed(self, symbol: str, side: str) -> bool:
        now = time.time()
        last_t = self._last_signal_time.get(symbol, 0.0)
        last_side = self._last_signal_side.get(symbol, '')
        if last_side == side and (now - last_t) < self.cooldown_minutes * 60:
            return False
        return True

    # ---------- Offentlig API ----------

    def analyze(self, symbol: str,
                df_4h: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_15m: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """
        Returnerer et signal hvis alle filtre er passert, ellers None.
        Forventer OHLCV med kolonner: ['open','high','low','close','volume'].
        """
        assert all(col in df_1h.columns for col in ['open','high','low','close','volume']), "1H mangler kolonner"
        assert all(col in df_4h.columns for col in ['open','high','low','close','volume']), "4H mangler kolonner"

        regime = self._regime(df_4h)
        adx1h = self._adx(df_1h).iloc[-1]
        atr1h = self._atr(df_1h).iloc[-1]
        price = float(df_1h['close'].iloc[-1])

        # Volum-bekreftelse
        if not self._vol_ok(df_1h):
            return None

        # Retningsfilter + ADX-gating
        side = None
        if regime in ('trend_up','strong_trend_up'):
            if adx1h >= self.regime_4h_adx_th:
                side = 'long'
        elif regime in ('trend_down','strong_trend_down'):
            if adx1h >= self.regime_4h_adx_th:
                side = 'short'
        else:  # range
            # enkel mean-reversion: mot ytterpunkter (kan toggles i config senere)
            ema_mid = self._ema(df_1h['close'], 50).iloc[-1]
            if price < ema_mid and adx1h < self.regime_4h_adx_th:
                side = 'long'
            elif price > ema_mid and adx1h < self.regime_4h_adx_th:
                side = 'short'

        if side is None:
            return None

        # Mikro-bekreftelse fra 15m om tilgjengelig
        if df_15m is not None and len(df_15m) >= 50:
            if not self._micro_conf(df_15m, side):
                return None

        # Anti-spam / cooldown
        if not self._cooldown_passed(symbol, side):
            return None

        # SL/TP
        sl = price - self.atr_mult_sl * atr1h if side == 'long' else price + self.atr_mult_sl * atr1h
        r = abs(price - sl)
        tp1 = price + self.tp_r1 * r if side == 'long' else price - self.tp_r1 * r
        tp2 = price + self.tp_r2 * r if side == 'long' else price - self.tp_r2 * r

        sig = Signal(
            symbol=symbol,
            side=side,
            price=price,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            meta={
                'regime': regime,
                'adx1h': float(adx1h),
                'atr1h': float(atr1h),
            }
        )
        # registrer cooldown
        self._last_signal_time[symbol] = time.time()
        self._last_signal_side[symbol] = side
        return sig
