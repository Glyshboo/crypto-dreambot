import pandas as pd
import numpy as np
from typing import Dict, Any
from indicators import atr
from strategy import TrendBias, compute_trend_bias, pullback_signal, breakout_signal, build_order_plan

def simulate_trades(df4h: pd.DataFrame, df1h: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Simple event-driven backtest on hourly bars using our rules and ATR exits."""
    records = []
    df1h = df1h.copy()
    atr_series = atr(df1h, cfg["trailing"]["atr_period"])

    for i in range(300, len(df1h)-1):
        ts = df1h.index[i]
        df4h_subset = df4h[df4h.index <= ts]
        if len(df4h_subset) < 50:
            continue

        bias = compute_trend_bias(df4h_subset, cfg)
        if bias == TrendBias.FLAT:
            continue

        pb = pullback_signal(df1h.iloc[:i+1], bias, cfg)
        bo = breakout_signal(df1h.iloc[:i+1], bias, cfg)
        if not pb and not bo:
            continue

        plan = build_order_plan(df1h.iloc[:i+1], bias, cfg)
        if plan is None:
            continue

        entry_price = float(df1h["open"].iloc[i+1])
        stop = float(plan["stop"])
        tp_partial = float(plan["tp_partial"])
        trail_mult = float(plan["trail_atr_mult"])

        long_side = (bias == TrendBias.LONG)
        direction = 1 if long_side else -1

        took_partial = False
        pnl_r = 0.0
        rdist = abs(entry_price - stop) if entry_price != stop else 1e-6

        for j in range(i+1, len(df1h)):
            h = float(df1h["high"].iloc[j])
            l = float(df1h["low"].iloc[j])
            atr_j = float(atr_series.iloc[j]) if not np.isnan(atr_series.iloc[j]) else 0.0

            # Stop check
            if long_side and l <= stop:
                pnl_r += -1.0 if not took_partial else 0.0
                break
            if (not long_side) and h >= stop:
                pnl_r += -1.0 if not took_partial else 0.0
                break

            # Partial at +1.8R
            target = tp_partial
            if not took_partial:
                if (long_side and h >= target) or ((not long_side) and l <= target):
                    took_partial = True
                    stop = entry_price  # move to breakeven
                    pnl_r += 0.4 * 1.8

            # Trailing
            if took_partial and atr_j > 0:
                trail = entry_price + direction * trail_mult * atr_j
                if long_side:
                    stop = max(stop, trail)
                else:
                    stop = min(stop, trail)

        records.append({"timestamp": df1h.index[i+1], "bias": bias, "entry": entry_price, "pnl_r": pnl_r})

    return pd.DataFrame(records)
