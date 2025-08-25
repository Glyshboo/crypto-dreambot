from typing import Dict, Any, Optional
from binance.um_futures import UMFutures
from data_feed import get_funding_rate

class ExecutionEngine:
    def __init__(self, client: UMFutures, cfg: Dict[str, Any], notifier=None):
        self.client = client
        self.cfg = cfg
        self.notifier = notifier

    def set_leverage(self, symbol: str, leverage: int = 5):
        lev = min(max(1, leverage), self.cfg["leverage"]["max"])
        try:
            self.client.change_leverage(symbol=symbol, leverage=lev)
            if self.notifier: self.notifier.send(f"{symbol} leverage set to {lev}x")
        except Exception as e:
            msg = f"[WARN] set_leverage failed for {symbol}: {e}"
            print(msg)
            if self.notifier: self.notifier.send(msg)

    def funding_ok(self, symbol: str, bias: str, stretch: bool) -> str:
        fcfg = self.cfg.get("funding_filter", {})
        if not fcfg.get("enabled", True):
            return "ok"
        fr = get_funding_rate(self.client, symbol)
        if bias == "long" and fr >= fcfg["threshold_pos"] and stretch:
            return "block_or_reduce"
        if bias == "short" and fr <= fcfg["threshold_neg"] and stretch:
            return "block_or_reduce"
        return "ok"

    def place_limit(self, symbol: str, side: str, qty: float, price: float) -> Optional[dict]:
        try:
            res = self.client.new_order(symbol=symbol, side=side.upper(), type="LIMIT",
                                         quantity=qty, price=price, timeInForce="GTC")
            if self.notifier: self.notifier.send(f"{symbol} {side} LIMIT placed: qty={qty}, price={price}")
            return res
        except Exception as e:
            msg = f"[ERROR] limit order failed: {e}"
            print(msg)
            if self.notifier: self.notifier.send(msg)
            return None

    def place_market(self, symbol: str, side: str, qty: float) -> Optional[dict]:
        try:
            res = self.client.new_order(symbol=symbol, side=side.upper(), type="MARKET", quantity=qty)
            if self.notifier: self.notifier.send(f"{symbol} {side} MARKET placed: qty={qty}")
            return res
        except Exception as e:
            msg = f"[ERROR] market order failed: {e}"
            print(msg)
            if self.notifier: self.notifier.send(msg)
            return None
