
from typing import Dict, Any, List, Optional
from binance.um_futures import UMFutures
import time

RECV_WINDOW = 5000  # ms

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def server_drift_ms(client: UMFutures) -> Optional[float]:
    try:
        st = client.time()
        server_ms = float(st.get('serverTime', 0))
        local_ms = time.time() * 1000.0
        return local_ms - server_ms
    except Exception:
        return None

def get_usdc_futures_balance(client: UMFutures) -> float:
    try:
        bals = client.balance(recvWindow=RECV_WINDOW)
        for b in bals:
            if b.get("asset") == "USDC":
                for k in ("availableBalance", "balance", "walletBalance"):
                    if k in b:
                        return _safe_float(b.get(k, 0.0))
    except Exception as e:
        print(f"[WARN] balance check failed: {e}")
    return 0.0

def get_symbol_settings(client: UMFutures, symbol: str) -> Dict[str, Any]:
    info = {"symbol": symbol, "leverage": None, "marginType": None}
    try:
        acc = client.account(recvWindow=RECV_WINDOW)
        positions = acc.get("positions", [])
        for p in positions:
            if p.get("symbol") == symbol:
                lev = p.get("leverage", 0)
                try:
                    lev = int(float(lev))
                except Exception:
                    lev = 0
                iso = p.get("isolated")
                iso_flag = False
                if isinstance(iso, bool):
                    iso_flag = iso
                elif isinstance(iso, str):
                    iso_flag = iso.lower() == "true"
                info["leverage"] = lev
                info["marginType"] = "isolated" if iso_flag else "cross"
                return info
    except Exception as e:
        print(f"[WARN] account() positions fetch failed for {symbol}: {e}")
    return info

def startup_check(client: UMFutures, symbols: List[str], expect_isolated: bool = True, max_leverage: int = 15, min_usdc_balance: float = 50.0) -> Dict[str, Any]:
    report = {"symbols": [], "balance_ok": False, "usdc_balance": 0.0, "drift_ms": None}
    report["drift_ms"] = server_drift_ms(client)
    bal = get_usdc_futures_balance(client)
    report["usdc_balance"] = bal
    report["balance_ok"] = bal >= min_usdc_balance
    for sym in symbols:
        s = get_symbol_settings(client, sym)
        ok_isolated = (s.get("marginType") == ("isolated" if expect_isolated else "cross"))
        lev = s.get("leverage") or 0
        ok_lev = (lev >= 1) and (lev <= max_leverage)
        s["ok_isolated"] = ok_isolated if s.get("marginType") is not None else None
        s["ok_leverage"] = ok_lev if s.get("leverage") is not None else None
        report["symbols"].append(s)
    return report
