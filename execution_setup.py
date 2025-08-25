from typing import Tuple, Dict, Any
from binance.um_futures import UMFutures

RECV_WINDOW = 5000

def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    try:
        return bool(v)
    except Exception:
        return False

def get_symbol_flags(client: UMFutures, symbol: str, recvWindow: int = RECV_WINDOW) -> Tuple[str, int]:
    """
    Returnerer (marginType, leverage) for symbolet, f.eks. ("ISOLATED", 12)
    """
    acc = client.account(recvWindow=recvWindow)
    for p in acc.get("positions", []):
        if p.get("symbol") == symbol:
            lev = int(float(p.get("leverage", 0) or 0))
            iso = _to_bool(p.get("isolated"))
            return ("ISOLATED" if iso else "CROSSED", lev)
    return ("UNKNOWN", 0)

def is_flat_position(client: UMFutures, symbol: str, recvWindow: int = RECV_WINDOW) -> bool:
    """
    Sjekker at posisjonen er flat (qty = 0) før vi endrer leverage.
    """
    try:
        acc = client.account(recvWindow=recvWindow)
        for p in acc.get("positions", []):
            if p.get("symbol") == symbol:
                amt = float(p.get("positionAmt") or 0.0)
                return abs(amt) == 0.0
    except Exception:
        pass
    return True  # fall-back: anta flat hvis vi ikke får svar

def set_isolated(client: UMFutures, symbol: str, want_iso: bool, recvWindow: int = RECV_WINDOW):
    """
    Setter margin type (ISOLATED/CROSSED).
    """
    want = "ISOLATED" if want_iso else "CROSSED"
    try:
        client.change_margin_type(symbol=symbol, marginType=want, recvWindow=recvWindow)
        return True, f"Margin satt til {want}"
    except Exception as e:
        msg = str(e)
        if "No need to change" in msg or "no need to change" in msg:
            return True, f"Margin allerede {want}"
        return False, f"Margin set feilet: {msg}"

def set_leverage(client: UMFutures, symbol: str, lev: int, recvWindow: int = RECV_WINDOW):
    """
    Setter ønsket leverage (1..max for symbolet).
    """
    lev = max(1, int(lev))
    try:
        client.change_leverage(symbol=symbol, leverage=lev, recvWindow=recvWindow)
        return True, f"Leverage satt til {lev}x"
    except Exception as e:
        return False, f"Leverage set feilet: {e}"

def safe_set_symbol_leverage(client: UMFutures, symbol: str, desired_leverage: int, recvWindow: int = RECV_WINDOW):
    """
    Trygg måte å sette leverage: endrer KUN når posisjon er flat.
    """
    try:
        if not is_flat_position(client, symbol, recvWindow):
            return False, "Leverage hoppa over (åpen posisjon)"
        ok, msg = set_leverage(client, symbol, int(desired_leverage), recvWindow)
        return ok, msg
    except Exception as e:
        return False, f"Leverage set feilet: {e}"

def autofix_symbol(client: UMFutures, symbol: str, desired_margin: str = "ISOLATED", desired_leverage: int = 12, recvWindow: int = RECV_WINDOW) -> Dict[str, Any]:
    """
    Kalles ved oppstart i LIVE:
      - Setter ISOLATED (om ønsket)
      - Setter leverage (f.eks. 12x)
    Returnerer en liten rapport for logging/Telegram.
    """
    current_mt, current_lev = get_symbol_flags(client, symbol, recvWindow=recvWindow)
    want_iso = (desired_margin.upper() == "ISOLATED")
    res = {"symbol": symbol, "before": {"margin": current_mt, "leverage": current_lev}, "after": {}, "actions": []}

    # Margin type
    if current_mt != desired_margin.upper():
        ok_m, msg_m = set_isolated(client, symbol, want_iso, recvWindow)
    else:
        ok_m, msg_m = True, "Ingen endring (margin)"
    res["actions"].append(msg_m)

    # Leverage
    if current_lev != int(desired_leverage) and current_lev != 0:
        ok_l, msg_l = set_leverage(client, symbol, int(desired_leverage), recvWindow)
    else:
        ok_l, msg_l = True, "Ingen endring (leverage)"
    res["actions"].append(msg_l)

    after_mt, after_lev = get_symbol_flags(client, symbol, recvWindow=recvWindow)
    res["after"] = {"margin": after_mt, "leverage": after_lev}
    res["ok"] = ok_m and ok_l and (after_mt == desired_margin.upper()) and (after_lev == int(desired_leverage) or after_lev == 0)
    return res


# === Added helpers: exchange filters, normalization, and margin-capping ===
import time
from binance.um_futures import UMFutures

_EXINFO_CACHE = {"ts": 0, "data": None}
_EXINFO_TTL = 12*60*60  # 12 hours

def _exchange_info_cached(client: UMFutures):
    now = int(time.time())
    if _EXINFO_CACHE["data"] is None or (now - _EXINFO_CACHE["ts"]) > _EXINFO_TTL:
        try:
            _EXINFO_CACHE["data"] = client.exchange_info()
            _EXINFO_CACHE["ts"] = now
        except Exception:
            _EXINFO_CACHE["data"] = None
    return _EXINFO_CACHE["data"]

_SYMBOL_FILTERS = {}

def get_symbol_filters(client: UMFutures, symbol: str):
    if symbol in _SYMBOL_FILTERS:
        return _SYMBOL_FILTERS[symbol]
    ex = _exchange_info_cached(client) or client.exchange_info()
    s = next((x for x in ex.get("symbols", []) if x.get("symbol")==symbol), None)
    if not s:
        raise ValueError(f"Symbol {symbol} not found")
    f = {f["filterType"]: f for f in s.get("filters", [])}
    tick = float(f["PRICE_FILTER"]["tickSize"])
    step = float(f["LOT_SIZE"]["stepSize"])
    min_qty = float(f["LOT_SIZE"]["minQty"])
    min_notional = float(f.get("MIN_NOTIONAL", {}).get("notional", 0.0)) if "MIN_NOTIONAL" in f else 0.0
    _SYMBOL_FILTERS[symbol] = {"tick": tick, "step": step, "min_qty": min_qty, "min_notional": min_notional}
    return _SYMBOL_FILTERS[symbol]

def _round_floor(value: float, quantum: float) -> float:
    if quantum <= 0: return float(value)
    return float(int(float(value)/quantum) * quantum)

def normalize_price_qty(client: UMFutures, symbol: str, price: float, qty: float):
    flt = get_symbol_filters(client, symbol)
    p = _round_floor(float(price), flt["tick"])
    q = _round_floor(float(qty),  flt["step"])
    if q < flt["min_qty"]:
        q = flt["min_qty"]
    if flt["min_notional"] > 0 and (p*q) < flt["min_notional"]:
        q = _round_floor(flt["min_notional"]/max(p,1e-9), flt["step"])
    return p, q

def max_qty_from_margin(balance_usdc: float, leverage: float, price: float, safety: float = 0.98) -> float:
    if price <= 0: return 0.0
    notional_cap = balance_usdc * max(1.0, float(leverage)) * float(safety)
    return max(0.0, notional_cap / price)
