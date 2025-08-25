
import json, threading, time
from collections import defaultdict, deque

try:
    # binance-connector >=3.x
    from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient as _WSClient
except Exception:
    try:
        # alt path some envs
        from binance.websocket.um_futures import UMFuturesWebsocketClient as _WSClient
    except Exception:
        _WSClient = None

try:
    import pandas as pd
except Exception:
    pd = None

class WSManager:
    """
    Optional WS manager for UM Futures:
    - Subscribes klines for symbols/intervals (e.g., 15m,1h)
    - Subscribes mark price
    - Keeps small in-memory ring buffers of closed candles
    - get_df(symbol, interval, limit) -> pandas.DataFrame or None
    """
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        self._client = None
        self._thread = None
        self._running = False
        self._buffers = defaultdict(lambda: deque(maxlen=600))  # key=(symbol, interval) -> deque of dicts
        self._marks = {}  # symbol -> last mark price
        self._lock = threading.Lock()
        self._ready = False
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

    @property
    def available(self):
        return _WSClient is not None and pd is not None

    @property
    def ready(self):
        return self._ready

    def _on_message(self, msg):
        try:
            if isinstance(msg, (bytes, bytearray)):
                msg = msg.decode("utf-8", errors="ignore")
            if isinstance(msg, str):
                data = json.loads(msg)
            else:
                data = msg
        except Exception:
            return

        # Normalize shapes
        try:
            # Mark price stream
            if "event" in data and data.get("event") in ("markPriceUpdate","markPriceUpdateEvent"):
                sym = data.get("symbol") or data.get("s")
                price = float(data.get("markPrice") or data.get("p"))
                with self._lock:
                    self._marks[sym] = price
                return
            if data.get("e") in ("markPriceUpdate",):
                sym = data.get("s")
                price = float(data.get("p"))
                with self._lock:
                    self._marks[sym] = price
                return
        except Exception:
            pass

        # Kline variations
        k = None
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict) and ("k" in data["data"] or "kline" in data["data"]):
                k = data["data"].get("k") or data["data"].get("kline")
            elif "k" in data:
                k = data["k"]
            elif "kline" in data:
                k = data["kline"]

        if not k:
            return

        try:
            sym = k.get("s") or k.get("symbol")
            interval = k.get("i") or k.get("interval")
            is_closed = bool(k.get("x"))
            ts_open = int(k.get("t") or k.get("startTime") or 0)
            ts_close = int(k.get("T") or k.get("closeTime") or 0)
            o = float(k.get("o")); h = float(k.get("h")); l = float(k.get("l")); c = float(k.get("c"))
            v = float(k.get("v") or 0.0)
        except Exception:
            return

        if not sym or not interval:
            return

        if is_closed:
            bar = {"open_time": ts_open, "close_time": ts_close, "open": o, "high": h, "low": l, "close": c, "volume": v}
            key = (sym, interval)
            with self._lock:
                dq = self._buffers[key]
                if len(dq) and dq[-1]["close_time"] == bar["close_time"]:
                    dq[-1] = bar
                else:
                    dq.append(bar)
                self._ready = True

    def start(self, symbols, intervals=("15m","1h"), mark_price=True):
        if not self.available:
            return False
        if self._running:
            return True
        self._running = True

        def _run():
            self._client = _WSClient()
            # subscribe klines
            for itv in intervals:
                for s in symbols:
                    try:
                        self._client.kline(
                            symbol=s.lower(),
                            id=f"k_{s}_{itv}",
                            interval=itv,
                            callback=self._on_message
                        )
                    except Exception:
                        pass
            if mark_price:
                for s in symbols:
                    try:
                        self._client.mark_price(
                            symbol=s.lower(),
                            id=f"mp_{s}",
                            speed=1,
                            callback=self._on_message
                        )
                    except Exception:
                        pass
            # keep thread alive
            while self._running:
                time.sleep(0.5)
            try:
                self._client.stop()
            except Exception:
                pass

        self._thread = threading.Thread(target=_run, name="WSManager", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False

    def get_df(self, symbol: str, interval: str, limit: int = 400):
        if pd is None:
            return None
        key = (symbol, interval)
        with self._lock:
            bars = list(self._buffers.get(key, []))
        if not bars:
            return None
        bars = bars[-limit:]
        # Build DataFrame
        df = pd.DataFrame(bars)
        return df

    def last_mark(self, symbol: str):
        with self._lock:
            return self._marks.get(symbol)
