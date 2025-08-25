# -*- coding: utf-8 -*-
"""
main.py — Patch 02 (paper/backtest klar; live disabled)
- Robust oppstart
- Henter OHLCV-data fra CSV eller bruker dummy (for test)
- Kjører strategi og sender varsler via Notifier
"""
import os
import sys
import time
from typing import Dict, Any, Optional

import pandas as pd

from strategy import Strategy
from notifier import Notifier
from execution_setup import ExecutionSetup

def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    need = {'open','high','low','close','volume'}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV mangler kolonner: {need - set(df.columns)}")
    return df

def _dummy_df(n: int = 400) -> pd.DataFrame:
    import numpy as np
    close = pd.Series(100 + np.cumsum(np.random.randn(n)))
    high = close + np.random.rand(n) * 2.0
    low = close - np.random.rand(n) * 2.0
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(np.random.randint(100, 1000, size=n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})

def run_once(symbol: str,
             df_4h: pd.DataFrame,
             df_1h: pd.DataFrame,
             df_15m: Optional[pd.DataFrame],
             strategy: Strategy,
             notifier: Notifier,
             exec_setup: ExecutionSetup):
    sig = strategy.analyze(symbol, df_4h, df_1h, df_15m)
    if sig is None:
        return
    info = exec_setup.prepare_symbol(symbol)
    if not info.get("ok", False):
        return
    text = (f"📈 <b>{symbol}</b> {sig.side.upper()} @ {sig.price:.2f}\n"
            f"SL: {sig.sl:.2f} | TP1: {sig.tp1:.2f} | TP2: {sig.tp2:.2f}\n"
            f"Regime: {sig.meta.get('regime')} | ADX1h: {sig.meta.get('adx1h'):.1f} | ATR1h: {sig.meta.get('atr1h'):.3f}")
    notifier.send(text)

def main():
    symbol = os.getenv("SYMBOL", "SOLUSDT")
    mode = os.getenv("MODE", "paper")   # paper | backtest (live kommer i Patch 03)
    data_4h = os.getenv("CSV_4H", "")
    data_1h = os.getenv("CSV_1H", "")
    data_15m = os.getenv("CSV_15M", "")

    df_4h = _load_csv(data_4h) or _dummy_df()
    df_1h = _load_csv(data_1h) or _dummy_df()
    df_15m = _load_csv(data_15m) or None

    strategy = Strategy()
    notifier = Notifier(cooldown_sec=int(os.getenv("TG_COOLDOWN_SEC", "30")))
    exec_setup = ExecutionSetup(paper=(mode != "live"))

    if mode == "backtest":
        # enkel loop for demonstrasjon; full backtest flyttes til egen modul i Patch 03
        for i in range(300, len(df_1h)):
            window_1h = df_1h.iloc[:i].copy()
            window_4h = df_4h.iloc[:max(100, i//4)].copy()
            window_15m = df_15m.iloc[:i*4].copy() if df_15m is not None else None
            run_once(symbol, window_4h, window_1h, window_15m, strategy, notifier, exec_setup)
        return

    # paper (default): kjør i rolig loop
    interval_sec = float(os.getenv("LOOP_SEC", "60"))
    print(f"[main] Starter i {mode} mode for {symbol}. Loop={interval_sec}s")
    while True:
        try:
            run_once(symbol, df_4h, df_1h, df_15m, strategy, notifier, exec_setup)
        except KeyboardInterrupt:
            print("Avslutter...")
            break
        except Exception as e:
            print(f"[main] Feil: {e}")
        time.sleep(interval_sec)

if __name__ == "__main__":
    main()
