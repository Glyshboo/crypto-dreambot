# paper.py — Step 2: two-step partial exits + BE after first partial

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from datetime import datetime
import math

@dataclass
class Trade:
    symbol: str
    side: str           # "BUY" (long) / "SELL" (short)
    qty: float
    entry: float
    stop: float
    tp1: float
    rdist: float
    trail_mult: float
    risk_usdc: float
    opened_ts: float
    took_partial1: bool = False
    took_partial2: bool = False
    breakeven: bool = False
    pnl_r_locked: float = 0.0

class PaperBroker:
    def __init__(self, fees_cfg: Dict[str, Any], notifier=None, cfg: Dict[str, Any] = None):
        self.fees = fees_cfg or {}
        self.notifier = notifier
        self.cfg = cfg or {}
        self.state = {
            "balance": float(self.cfg.get("paper", {}).get("starting_balance", 1000.0)),
            "open_trades": [],  # list[Trade]
            "closed": [],       # list[dict]
            "daily": {}         # date -> dict
        }

    # ---- Public API used by main.py ----
    def open_trade(self, symbol: str, side: str, qty: float, entry: float, stop: float, tp1: float,
                   r_distance: float, trail_atr_mult: float, risk_usdc: float, now_ts: float | None = None):
        tr = Trade(
            symbol=symbol, side=side, qty=float(qty), entry=float(entry), stop=float(stop), tp1=float(tp1),
            rdist=float(r_distance) if r_distance else abs(entry - stop),
            trail_mult=float(trail_atr_mult), risk_usdc=float(risk_usdc), opened_ts=float(now_ts or 0.0)
        )
        self.state["open_trades"].append(tr)

    def on_bar(self, symbol: str, o: float, h: float, l: float, c: float, atr: float):
        still_open: List[Trade] = []
        for tr in self.state["open_trades"]:
            if tr.symbol != symbol:
                still_open.append(tr)
                continue

            long_side = (tr.side == "BUY")
            # SL hit?
            if long_side and l <= tr.stop:
                pnl_r = -1.0 if not tr.took_partial1 else 0.0
                self._close_trade(tr, "SL", pnl_r, exit_price=tr.stop, atr=atr)
                continue
            if (not long_side) and h >= tr.stop:
                pnl_r = -1.0 if not tr.took_partial1 else 0.0
                self._close_trade(tr, "SL", pnl_r, exit_price=tr.stop, atr=atr)
                continue

            # Multi-partials from cfg
            partials = self.cfg.get("take_profit", {}).get("partials", [
                {"r": 1.5, "pct": 0.40},
                {"r": 2.5, "pct": 0.30}
            ])
            rdist = tr.rdist or 1e-9
            price_r = (h - tr.entry) / rdist if long_side else (tr.entry - l) / rdist

            # First partial
            if not tr.took_partial1:
                r1 = float(partials[0].get("r", 1.5))
                if price_r >= r1:
                    tr.took_partial1 = True
                    tr.stop = tr.entry  # BE
                    tr.breakeven = True
                    tr.pnl_r_locked += float(partials[0].get("pct", 0.40)) * r1
                    if self.notifier:
                        try:
                            self.notifier._send(f"🟢 {symbol} partial 1 filled @ {r1:.2f}R – SL→BE")
                        except Exception:
                            pass

            # Second partial
            if tr.took_partial1 and not tr.took_partial2 and len(partials) > 1:
                r2 = float(partials[1].get("r", 2.5))
                if price_r >= r2:
                    tr.took_partial2 = True
                    tr.pnl_r_locked += float(partials[1].get("pct", 0.30)) * r2
                    if self.notifier:
                        try:
                            self.notifier._send(f"🟢 {symbol} partial 2 filled @ {r2:.2f}R")
                        except Exception:
                            pass

            # Trailing etter første partial
            if tr.took_partial1 and atr > 0:
                direction = 1 if long_side else -1
                candidate = tr.entry + direction * tr.trail_mult * atr
                tr.stop = max(tr.stop, candidate) if long_side else min(tr.stop, candidate)

            still_open.append(tr)

        self.state["open_trades"] = still_open

    def daily_report_hype(self, tz_name: str = "Europe/Oslo"):
        # Summerer dagens lukkede handler
        d = datetime.now().date().isoformat()
        rep = self.state["daily"].get(d, {"wins": 0, "losses": 0, "pnl_usdc": 0.0, "n": 0})
        wins = rep["wins"]; losses = rep["losses"]; n = rep["n"]
        wr = (wins / n * 100.0) if n > 0 else 0.0
        avg_r = rep.get("avg_r", 0.0)
        return d, wins, losses, wr, rep["pnl_usdc"], avg_r, n

    # ---- Internals ----
    def _close_trade(self, tr: Trade, reason: str, pnl_r: float, exit_price: float, atr: float):
        # R-locking er allerede oppdatert via partials; total PnL_R = locked + endelig (hvis SL/TP/exit)
        pnl_r_total = tr.pnl_r_locked + pnl_r
        usdc = pnl_r_total * max(1e-9, tr.risk_usdc)
        self.state["balance"] += usdc

        # Stats/dag
        d = datetime.now().date().isoformat()
        rep = self.state["daily"].setdefault(d, {"wins": 0, "losses": 0, "pnl_usdc": 0.0, "n": 0, "sum_r": 0.0})
        if pnl_r_total >= 0:
            rep["wins"] += 1
        else:
            rep["losses"] += 1
        rep["pnl_usdc"] += usdc
        rep["n"] += 1
        rep["sum_r"] += pnl_r_total
        rep["avg_r"] = rep["sum_r"] / max(1, rep["n"])

        self.state["closed"].append({
            "symbol": tr.symbol, "side": tr.side, "reason": reason,
            "pnl_r": pnl_r_total, "pnl_usdc": usdc, "exit": exit_price
        })
