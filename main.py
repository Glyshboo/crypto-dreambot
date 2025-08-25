# --- Optional WebSocket helper (safe) ---
ws_mgr = None
# === Re-entry cooldown (vedvarende) ===
import time, json as _json, os as _os
_COOLDOWN_FILE = _os.path.join(_os.path.dirname(__file__), "reentry_state.json")
try:
    with open(_COOLDOWN_FILE, "r", encoding="utf-8") as _f:
        _REENTRY_UNTIL = {k:int(v) for k,v in _json.load(_f).items()}
except Exception:
    _REENTRY_UNTIL = {}

_HAD_OPEN_LAST = {}  # per-symbol posisjonsstatus fra forrige loop

# === Re-entry cooldown (vedvarende) ===
import time, json as _json, os as _os
_COOLDOWN_FILE = _os.path.join(_os.path.dirname(__file__), "reentry_state.json")
try:
    with open(_COOLDOWN_FILE, "r", encoding="utf-8") as _f:
        _REENTRY_UNTIL = {k:int(v) for k,v in _json.load(_f).items()}
except Exception:
    _REENTRY_UNTIL = {}

_HAD_OPEN_LAST = {}  # per-symbol posisjonsstatus fra forrige loop

# === Parallell-posisjoner og volatilitet (ATR%%) helpers ===
# === Funding + Margin/Leverage guard helpers ===
_SYMBOL_INFO_CACHE = {}
_ENSURED_LEVERAGE_MARGIN = set()

# === Supertrend-trail helpers (vedvarende R0 sporing) ===
import json as _json, os as _os
_TRAIL_FILE = _os.path.join(_os.path.dirname(__file__), "trail_state.json")
try:
    with open(_TRAIL_FILE, "r", encoding="utf-8") as _f:
        _TRAIL_STATE = _json.load(_f)
except Exception:
    _TRAIL_STATE = {}

def _trail_get_R0(sym: str):
    try:
        return float(_TRAIL_STATE.get(sym, {}).get("R0", 0.0)) or None
    except Exception:
        return None

def _trail_set_R0(sym: str, R0: float):
    try:
        d = _TRAIL_STATE.get(sym, {}) or {}
        d["R0"] = float(R0)
        _TRAIL_STATE[sym] = d
        with open(_TRAIL_FILE, "w", encoding="utf-8") as _f:
            _json.dump(_TRAIL_STATE, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _atr_value(df, period=14):
    try:
        import pandas as _pd
        if df is None or len(df) < period + 2:
            return None
        d = df.copy()
        H, L, C = d["high"], d["low"], d["close"]
        tr = _pd.concat([(H - L).abs(), (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return float(atr.iloc[-1])
    except Exception:
        return None

def _get_open_stop_price(client, symbol: str):
    """Finn aktiv STOP_MARKET closePosition for symbol; returner (orderId, stopPrice) eller (None, None)."""
    try:
        oo = client.get_open_orders(symbol=symbol)
        for o in oo:
            if str(o.get("type","")).upper() == "STOP_MARKET":
                # sjekk at dette er en close-order (reduceOnly/closePosition)
                sp = o.get("stopPrice") or o.get("stopprice") or o.get("stop_price")
                if sp is not None:
                    return o.get("orderId") or o.get("orderID") or o.get("order_id"), float(sp)
    except Exception:
        pass
    return None, None

def _position_entry_side(client, symbol: str):
    """Returner (entryPrice, side_str) hvis posisjon åpen, ellers (None, None)."""
    try:
        # robust mot ulike klientversjoner
        try:
            pos = client.get_position_risk()
        except Exception:
            try:
                pos = client.position_risk()
            except Exception:
                pos = client.position_information()
        for p in pos:
            if p.get("symbol") == symbol:
                amt = float(p.get("positionAmt", 0) or 0)
                if abs(amt) > 0:
                    entry = float(p.get("entryPrice", 0) or 0)
                    side = "LONG" if amt > 0 else "SHORT"
                    return entry if entry>0 else None, side
    except Exception:
        pass
    return None, None

def _tighten_stop_if_better(client, cfg, symbol: str, pos_side: str, new_stop: float):
    """
    Stram SL ved å:
      1) kansellere eksisterende STOP_MARKET closePosition
      2) legge ny STOP_MARKET closePosition med høyere (LONG) / lavere (SHORT) stopPrice
    """
    try:
        tick, step = _exchange_filters(client, symbol)
    except Exception:
        try:
            flt = get_symbol_filters(client, symbol)
            tick, step = float(flt["tick"]), float(flt["step"])
        except Exception:
            tick, step = 0.0, 0.0

    def _round_tick(x, t):
        if t and t>0: 
            return round(round(float(x)/t)*t, 12)
        return float(x)

    # Finn eksisterende stop
    oid, cur_sp = _get_open_stop_price(client, symbol)
    # Sjekk forbedring
    if cur_sp is not None:
        if pos_side == "LONG" and not (new_stop > float(cur_sp)):
            return False
        if pos_side == "SHORT" and not (new_stop < float(cur_sp)):
            return False

    new_sp = _round_tick(new_stop, tick)
    stop_side = "SELL" if pos_side=="LONG" else "BUY"

    # Avbryt gammel order hvis vi fant en
    try:
        if oid is not None:
            client.cancel_order(symbol=symbol, orderId=int(oid))
    except Exception:
        pass

    # Legg ny STOP_MARKET (closePosition primært, fallback reduceOnly)
    try:
        _place_order_with_retries(client, symbol=symbol, side=stop_side, type="STOP_MARKET",
                                  workingType="MARK_PRICE", stopPrice=str(new_sp), closePosition=True)
    except Exception:
        _place_order_with_retries(client, symbol=symbol, side=stop_side, type="STOP_MARKET",
                                  workingType="MARK_PRICE", stopPrice=str(new_sp), reduceOnly="true")
    return True

def _entry_and_stop(client, symbol: str):
    e, _ = _position_entry_side(client, symbol)
    _, sp = _get_open_stop_price(client, symbol)
    return e, sp\n\ndef _exchange_info_symbol(client, symbol):
    try:
        if symbol in _SYMBOL_INFO_CACHE:
            return _SYMBOL_INFO_CACHE[symbol]
        ei = client.exchange_info()
        for s in ei.get("symbols", []):
            if s.get("symbol") == symbol:
                _SYMBOL_INFO_CACHE[symbol] = s
                return s
    except Exception:
        pass
    return {}

def _symbol_margin_asset(client, symbol):
    # USDⓈ-M futures: marginAsset er typisk "USDT" (noen "USDC").
    info = _exchange_info_symbol(client, symbol) or {}
    return info.get("marginAsset") or "USDT"

def _futures_balances(client):
    # Returner dict asset->free (USDⓈ-M)
    out = {}
    try:
        bals = client.balance()
    except Exception:
        try:
            bals = client.futures_account_balance()
        except Exception:
            bals = []
    try:
        for b in bals:
            asset = b.get("asset") or b.get("asset".lower())
            free = float(b.get("balance") or b.get("withdrawAvailable") or b.get("availableBalance") or b.get("availableBal") or b.get("free") or 0.0)
            if asset:
                out[asset] = max(out.get(asset, 0.0), free)
    except Exception:
        pass
    return out

def _ensure_leverage_margin(client, symbol, lev=12, marginType="ISOLATED"):
    # Kall bare én gang per symbol per prosess
    key = f"{symbol}|{lev}|{marginType}"
    if key in _ENSURED_LEVERAGE_MARGIN:
        return
    try:
        # margin type
        try:
            client.change_margin_type(symbol=symbol, marginType=marginType)
        except Exception:
            # -4046: No need to change, ignorer
            pass
        # leverage
        try:
            client.change_leverage(symbol=symbol, leverage=int(lev))
        except Exception:
            pass
    except Exception:
        pass
    _ENSURED_LEVERAGE_MARGIN.add(key)

def _premium_funding_rate(client, symbol):
    """
    Hent gjeldende funding via premiumIndex (har lastFundingRate) – fallback til funding_rate history.
    Returnerer float (per 8t), signert (+ betyr longs betaler).
    """
    try:
        px = client.premium_index(symbol=symbol)
        if isinstance(px, dict) and "lastFundingRate" in px:
            return float(px["lastFundingRate"])
    except Exception:
        pass
    try:
        fr = client.funding_rate(symbol=symbol, limit=1)
        if isinstance(fr, list) and fr:
            return float(fr[-1].get("fundingRate", 0.0))
    except Exception:
        pass
    return 0.0

def _dir_penalized(bias_dir, funding_rate, directional=True):
    """
    Long betaler ved positiv funding. Short betaler ved negativ funding.
    Hvis directional=False, straff alltid etter absoluttverdi.
    """
    if not directional:
        return abs(funding_rate), True
    d = (str(bias_dir) or "").upper()
    if d == "LONG" and funding_rate > 0:
        return abs(funding_rate), True
    if d == "SHORT" and funding_rate < 0:
        return abs(funding_rate), True
    return abs(funding_rate), False\n\ndef _count_open_positions(client):
    """
    Returnerer antall symbols med åpen posisjon (amt != 0).
    Robust mot ulike binance endepunkter.
    """
    try:
        # UMFutures position endpoints varierer med lib-versjon
        try:
            data = client.get_position_risk()
        except Exception:
            data = client.position_risk()
    except Exception:
        try:
            data = client.position_information()
        except Exception:
            data = []
    cnt = 0
    try:
        for p in data:
            amt = float(p.get("positionAmt") or p.get("positionAmt".lower()) or 0)
            if abs(amt) > 0:
                cnt += 1
    except Exception:
        pass
    return cnt

def _atr_percent(df, period=14):
    """
    ATR% = ATR / close. Bruker klassisk ATR (mean TR n-perioder).
    Forventer kolonner: high, low, close.
    """
    try:
        import pandas as _pd
        if df is None or len(df) < period + 2:
            return None
        d = df.copy()
        H, L, C = d["high"], d["low"], d["close"]
        tr = _pd.concat([(H - L).abs(), (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        last_close = float(C.iloc[-1])
        last_atr = float(atr.iloc[-1])
        if last_close <= 0:
            return None
        return last_atr / last_close
    except Exception:
        return None\n\ndef _cooldown_hours(cfg: dict) -> int:
    try:
        return int(cfg.get("portfolio", {}).get("cooldown_bars_after_exit", 2))
    except Exception:
        return 2

def _cooldown_active(sym: str) -> bool:
    return int(time.time()*1000) < int(_REENTRY_UNTIL.get(sym, 0))

def _cooldown_start(sym: str, hours: int):
    until = int(time.time()*1000) + int(hours)*3600*1000
    _REENTRY_UNTIL[sym] = until
    try:
        with open(_COOLDOWN_FILE, "w", encoding="utf-8") as _f:
            _json.dump(_REENTRY_UNTIL, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _cooldown_hours(cfg: dict) -> int:
    try:
        return int(cfg.get("portfolio", {}).get("cooldown_bars_after_exit", 2))
    except Exception:
        return 2

def _cooldown_active(sym: str) -> bool:
    return int(time.time()*1000) < int(_REENTRY_UNTIL.get(sym, 0))

def _cooldown_start(sym: str, hours: int):
    until = int(time.time()*1000) + int(hours)*3600*1000
    _REENTRY_UNTIL[sym] = until
    try:
        with open(_COOLDOWN_FILE, "w", encoding="utf-8") as _f:
            _json.dump(_REENTRY_UNTIL, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _maybe_start_ws(cfg, symbols, notifier):
    try:
        from ws_manager import WSManager
    except Exception:
        return None
    try:
        if not cfg.get('ws', {}).get('enabled', False):
            return None
        mgr = WSManager()
        if not mgr.available:
            try: notifier.warn_throttled('ws_na', 'WS-klient ikke tilgjengelig — bruker REST', 300)
            except Exception: pass
            return None
        ok = mgr.start(symbols, intervals=tuple(cfg.get('ws',{}).get('klines', ['15m','1h'])), mark_price=bool(cfg.get('ws',{}).get('mark_price', True)))
        if ok:
            try: notifier.info_throttled('ws_on', '🔌 WS aktivert (kline/markprice)', 300)
            except Exception: pass
            return mgr
        return None
    except Exception as e:
        try: notifier.warn_throttled('ws_err', f'WS init feilet: {e}', 300)
        except Exception: pass
        return None

import os, time, json
from strategy import should_emit, mark_emitted  # anti-spam
from dotenv import load_dotenv
from datetime import datetime
import pytz
from copy import deepcopy
import pandas as pd  # EMA20 for hybrid

from binance.um_futures import UMFutures
from data_feed import get_klines, get_funding_rate
from strategy import (
    compute_trend_bias, pullback_signal, breakout_signal, build_order_plan,
    TrendBias, compute_confidence, compute_market_score, Regime,
    volatility_guard, get_atr_price_threshold, mean_reversion_signal
)
from risk import choose_risk_usdc, position_qty
from notifier import Notifier
from paper import PaperBroker
from telecontrol import TeleControl
from binance_checks import startup_check, get_usdc_futures_balance
from execution_setup import autofix_symbol, safe_set_symbol_leverage, get_symbol_filters, normalize_price_qty, max_qty_from_margin

# ---------------- utils ----------------

def load_config(path: str="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_overrides(cfg: dict) -> dict:
    try:
        from runtime_overrides import load_overrides
        ov = load_overrides()
        if isinstance(ov, dict) and "risk" in ov:
            cfg["risk"].update(ov["risk"])
    except Exception:
        pass
    return cfg

def map_confidence_to_leverage(score: float, cfg):
    pol = cfg.get("leverage_policy", {})
    if score >= pol.get("high_threshold", 0.8):
        return pol.get("high", 15)
    if score <= pol.get("low_threshold", 0.3):
        return pol.get("low", 8)
    return pol.get("normal", 12)

def current_balance(paper_mode: bool, broker_sim: PaperBroker, client: UMFutures) -> float:
    if paper_mode:
        try:
            return float(broker_sim.state.get("balance", 1000.0))
        except Exception:
            return 1000.0
    else:
        return float(get_usdc_futures_balance(client) or 0.0)

def _closed_slice(df):
    if df is None or len(df) < 2: return None
    return df.iloc[:-1]

def _bar_id(df):
    if df is None or len(df) == 0: return 0
    if "close_time" in df.columns:
        try: return int(df["close_time"].iloc[-1])
        except Exception: pass
    return len(df)

def _fib_mid_from_entry_stop(side: str, entry: float, stop: float) -> float:
    if entry <= 0 or stop <= 0: return entry
    low, high = (stop, entry) if side == "BUY" else (entry, stop)
    span = max(1e-9, high - low)
    lvl382 = low + 0.382 * span
    lvl618 = low + 0.618 * span
    return (lvl382 + lvl618) / 2.0

def _ema20_last(df1h_closed):
    try:
        return pd.Series(df1h_closed["close"]).ewm(span=20, adjust=False).mean().iloc[-1]
    except Exception:
        return None

# --- Breakout nivå-helper (bruker samme definisjon som breakout_signal) ---
def _breakout_level(df1h_closed: pd.DataFrame, bias: str, lookback: int) -> float | None:
    if len(df1h_closed) < max(20, lookback+2):
        return None
    if bias == TrendBias.LONG:
        return float(df1h_closed["close"].rolling(lookback).max().iloc[-2])
    elif bias == TrendBias.SHORT:
        return float(df1h_closed["close"].rolling(lookback).min().iloc[-2])
    return None

# --- Daily loss limit helper (regime-adaptiv + valgfrie USD-gulv/tak) ---
def _daily_loss_limit_usd(balance: float, cfg: dict, regime_name: str | None) -> float:
    risk = cfg.get("risk", {})

    overrides = risk.get("daily_loss_limit_pct_overrides", {})
    base_pct = float(risk.get("daily_loss_limit_pct", 0.0) or 0.0)
    regime_key = (regime_name or "trend")
    pct = float(overrides.get(regime_key, base_pct) or base_pct)

    cap_pct = float(risk.get("max_balance_pct", 1.0) or 1.0)
    min_usd = float(risk.get("daily_loss_min_usd", 0.0) or 0.0)
    max_usd = float(risk.get("daily_loss_max_usd", 0.0) or 0.0)

    raw = (pct / 100.0) * (balance * cap_pct)
    if min_usd > 0:
        raw = max(raw, min_usd)
    if max_usd > 0:
        raw = min(raw, max_usd)

    return -float(raw)

# ---------------- Regime auto-overrides ----------------

_LAST_REGIME_NAME = None

def _regime_with_hysteresis(score: float) -> str:
    global _LAST_REGIME_NAME
    up_trend, down_trend = 35, 30
    up_strong, down_strong = 70, 60
    if _LAST_REGIME_NAME is None:
        _LAST_REGIME_NAME = "range" if score < up_trend else ("strong_trend" if score >= up_strong else "trend")
        return _LAST_REGIME_NAME
    cur = _LAST_REGIME_NAME
    if cur == "range":
        if score >= up_strong: _LAST_REGIME_NAME = "strong_trend"
        elif score >= up_trend: _LAST_REGIME_NAME = "trend"
    elif cur == "trend":
        if score >= up_strong: _LAST_REGIME_NAME = "strong_trend"
        elif score < down_trend: _LAST_REGIME_NAME = "range"
    elif cur == "strong_trend":
        if score < down_strong: _LAST_REGIME_NAME = "trend"
    return _LAST_REGIME_NAME

def apply_regime_overrides(eff: dict, regime_name: str) -> dict:
    try:
        if regime_name == "range":
            eff["trend_filter"]["adx_min"] = max(eff["trend_filter"].get("adx_min", 20), 25)
            eff["entries"]["breakout"]["lookback"] = max(eff["entries"]["breakout"].get("lookback", 10), 15)
            eff["entries"]["breakout"]["vol_boost_pct"] = max(eff["entries"]["breakout"].get("vol_boost_pct", 120), 130)
            eff["take_profit"]["partial_r"] = min(eff["take_profit"].get("partial_r", 1.8), 1.5)
            eff["trailing"]["atr_mult"] = min(eff["trailing"].get("atr_mult", 2.3), 2.0)
            eff["portfolio"]["max_concurrent"] = min(eff["portfolio"].get("max_concurrent", 2), 1)
            eff["portfolio"]["cooldown_bars"] = max(eff["portfolio"].get("cooldown_bars", 2), 3)
            eff["leverage_policy"]["normal"] = min(eff["leverage_policy"].get("normal", 12), 10)
            eff["leverage_policy"]["high"]   = min(eff["leverage_policy"].get("high", 15), 12)
        elif regime_name == "strong_trend":
            eff["trend_filter"]["adx_min"] = min(eff["trend_filter"].get("adx_min", 20), 20)
            eff["take_profit"]["partial_r"] = max(eff["take_profit"].get("partial_r", 1.8), 2.0)
            eff["trailing"]["atr_mult"] = max(eff["trailing"].get("atr_mult", 2.3), 2.8)
            eff["portfolio"]["max_concurrent"] = max(eff["portfolio"].get("max_concurrent", 2), 2)
            eff["funding_filter"]["threshold_pos"] = max(eff["funding_filter"].get("threshold_pos", 0.0005), 0.0008)
            eff["funding_filter"]["threshold_neg"] = min(eff["funding_filter"].get("threshold_neg", -0.0005), -0.0008)
    except Exception:
        pass
    return eff

# ---------------- LIVE helpers ----------------

def _exchange_filters(client: UMFutures, symbol: str):
    tick = 0.01; step = 0.001
    ex = client.exchange_info()
    for s in ex.get("symbols", []):
        if s.get("symbol")==symbol:
            for f in s.get("filters", []):
                if f.get("filterType")=="PRICE_FILTER":
                    tick = float(f.get("tickSize", tick))
                if f.get("filterType")=="LOT_SIZE":
                    step = float(f.get("stepSize", step))
            break
    return tick, step

def _round_step(x: float, step: float):
    if step <= 0: return x
    return (int(x/step))*step

def _round_tick(p: float, tick: float):
    if tick <= 0: return p
    return (int(p/tick))*tick


def _place_order_with_retries(client, notifier=None, max_retries=6, backoff_base=0.6, **kwargs):
    """
    Robust ordrefunksjon for Binance USDⓈ-M futures.
    Forventer kwargs som brukes av client.new_order(...):
        symbol, side, type, quantity, price, reduceOnly, closePosition, workingType, stopPrice, timeInForce, etc.
    - Eksponentiell backoff med jitter
    - Idempotent via newClientOrderId (samme COID ved retry)
    - Automatisk rounding mot tick/step ved -1013/precision
    - Fallback reduceOnly <-> closePosition for TP/SL
    - Skalerer qty ned ved insufficient margin
    """
    import time, uuid, random
    from math import floor

    # idempotens-lager
    _idem_file = os.path.join(os.path.dirname(__file__), "idempotency.json")
    try:
        import json as _json
        if os.path.exists(_idem_file):
            with open(_idem_file, "r", encoding="utf-8") as _f:
                _idem = _json.load(_f)
        else:
            _idem = {}
    except Exception:
        _idem = {}

    symbol = kwargs.get("symbol")
    side = kwargs.get("side")
    otype = kwargs.get("type", "MARKET")

    # legg til recvWindow hvis ikke satt
    kwargs.setdefault("recvWindow", 5000)

    # COID – lag stabil id for retries
    if "newClientOrderId" not in kwargs:
        base = f"DB-{symbol}-{side}-{otype}-{int(time.time()*1000)%100000000}"
        coid = base + "-" + uuid.uuid4().hex[:6]
        kwargs["newClientOrderId"] = coid
    else:
        coid = kwargs["newClientOrderId"]

    # Idempotens: hvis vi ser samme coid som "ferdig", returnér "suksess"
    if _idem.get(coid) == "ok":
        return {"clientOrderId": coid, "idempotent": True}

    # step/tick helpers
    def _get_filters(sym):
        try:
            flt = get_symbol_filters(client, sym)
            return float(flt["tick"]), float(flt["step"])
        except Exception:
            try:
                t, s = _exchange_filters(client, sym)
                return float(t), float(s)
            except Exception:
                return 0.0, 0.0

    def _round_to_step(val, step):
        if step <= 0: return float(val)
        return floor(float(val)/step)*step

    def _round_to_tick(val, tick):
        if tick <= 0: return float(val)
        return round(round(float(val)/tick)*tick, 12)

    tick, step = _get_filters(symbol)

    # sørg for str-kwargs der binance trenger str
    def _to_str_if_needed(k, v):
        if v is None: return v
        if k in ("quantity","price","stopPrice","icebergQty","activationPrice","callbackRate"):
            try: return str(v)
            except Exception: return v
        return v

    # hoved-retry loop
    last_err = None
    qty_key = "quantity"
    for attempt in range(1, int(max_retries)+1):
        # juster kvantum/price til exchange filtere om de finnes
        try:
            if qty_key in kwargs and kwargs[qty_key] is not None:
                qf = float(kwargs[qty_key])
                if step > 0:
                    qf = max(step, _round_to_step(qf, step))
                kwargs[qty_key] = qf
            if "price" in kwargs and kwargs["price"] is not None and tick > 0:
                kwargs["price"] = _round_to_tick(kwargs["price"], tick)
            if "stopPrice" in kwargs and kwargs["stopPrice"] is not None and tick > 0:
                kwargs["stopPrice"] = _round_to_tick(kwargs["stopPrice"], tick)
        except Exception:
            pass

        send_kwargs = {k: _to_str_if_needed(k, v) for k, v in kwargs.items()}

        try:
            # primærkall
            resp = client.new_order(**send_kwargs)
            # marker idem "ok"
            try:
                _idem[coid] = "ok"
                with open(_idem_file, "w", encoding="utf-8") as _f:
                    _json.dump(_idem, _f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return resp
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # kjente mønstre
            is_rate = ("status code: 429" in msg) or ("status code: 418" in msg) or ("too many" in msg)
            is_ts   = ("timestamp" in msg) or ("recvwindow" in msg) or ("-1021" in msg)
            is_prec = ("precision" in msg) or ("invalid quantity" in msg) or ("-1013" in msg) or ("min_notional" in msg) or ("lot size" in msg)
            is_margin = ("insufficient" in msg) or ("-2019" in msg)
            is_reduce = ("reduceonly" in msg) or ("-4164" in msg)
            is_exist  = ("exist" in msg) and ("order" in msg)

            # duplikat / already exists -> suksess
            if is_exist:
                try:
                    _idem[coid] = "ok"
                    with open(_idem_file, "w", encoding="utf-8") as _f:
                        _json.dump(_idem, _f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                return {"clientOrderId": coid, "accepted": "exists"}

            # precision -> prøv å korrigere tick/step
            if is_prec:
                try:
                    if qty_key in kwargs and kwargs[qty_key] is not None and step > 0:
                        kwargs[qty_key] = max(step, _round_to_step(float(kwargs[qty_key]), step))
                    if "price" in kwargs and kwargs.get("price") not in (None, "") and tick > 0:
                        kwargs["price"] = _round_to_tick(kwargs["price"], tick)
                    if "stopPrice" in kwargs and kwargs.get("stopPrice") not in (None, "") and tick > 0:
                        kwargs["stopPrice"] = _round_to_tick(kwargs["stopPrice"], tick)
                except Exception:
                    pass

            # insufficient margin -> skaler qty litt ned
            if is_margin and qty_key in kwargs and kwargs[qty_key] not in (None, ""):
                try:
                    qf = float(kwargs[qty_key])
                    qf = qf * 0.95  # -5%
                    if step > 0:
                        qf = max(step, _round_to_step(qf, step))
                    kwargs[qty_key] = qf
                except Exception:
                    pass

            # reduceOnly/closePosition konflikt -> flip mellom dem
            if is_reduce:
                ro = str(kwargs.get("reduceOnly", "")).lower()
                cp = kwargs.get("closePosition", None)
                if ro in ("true","1",True):
                    kwargs.pop("reduceOnly", None)
                    kwargs["closePosition"] = True
                elif cp is True:
                    kwargs.pop("closePosition", None)
                    kwargs["reduceOnly"] = "true"

            # timestamp/recvWindow -> øk window litt
            if is_ts:
                try:
                    rw = int(kwargs.get("recvWindow", 5000))
                    kwargs["recvWindow"] = min(60000, int(rw * 1.5))
                except Exception:
                    pass

            # backoff
            delay = backoff_base * (2 ** (attempt-1)) + random.random()*0.25
            try:
                if notifier: notifier.info(f"Retry order ({attempt}/{max_retries}) {symbol} {side} {otype}: {e} (sleep {delay:.2f}s)")
                else: print(f"[RETRY] {symbol} {side} {otype}: {e} (sleep {delay:.2f}s)")
            except Exception:
                pass
            time.sleep(delay)

    # alle forsøk feilet
    if notifier:
        try: notifier.error(f"Order FAIL {symbol} {side} {otype}: {last_err}")
        except Exception: pass
    raise last_err

def 
def _place_live_bracket(client: UMFutures, cfg: dict, broker, paper_mode: bool, notifier, symbol: str, side: str,
                        qty: float, entry: float, stop: float, tp1: float, partial_pct: float):
    """
    Plasser market-entry + STOP_MARKET + TP1 (+ TP2 hvis konfigurert).
    """
    # Filters & rounding
    try:
        flt = get_symbol_filters(client, symbol)
        tick = flt['tick']; step = flt['step']
    except Exception:
        tick, step = _exchange_filters(client, symbol)  # fallback
        flt = {'min_qty': step, 'min_notional': 0.0}

    # Cap mot margin & normaliser qty
    try:
        bal = current_balance(paper_mode, broker, client)
    except Exception:
        bal = 0.0
    lev = cfg.get('leverage', {}).get('base', 12)
    maxq = max_qty_from_margin(bal or 0.0, lev, entry, 0.97)
    qty = min(qty, maxq)
    _, qty_norm = normalize_price_qty(client, symbol, entry, qty)
    qty_r = max(step, _round_step(qty_norm, step))

    # MARKET entry
    _place_order_with_retries(client, symbol=symbol, side=side, type="MARKET", quantity=str(qty_r))

    # STOP (MARK_PRICE)
    side_sl = "SELL" if side=="BUY" else "BUY"
    stop_r = _round_tick(stop, tick)
    try:
        _place_order_with_retries(client, symbol=symbol, side=side_sl, type="STOP_MARKET",
                                  workingType="MARK_PRICE", stopPrice=str(stop_r), closePosition=True)
    except Exception:
        _place_order_with_retries(client, symbol=symbol, side=side_sl, type="STOP_MARKET",
                                  workingType="MARK_PRICE", stopPrice=str(stop_r), quantity=str(qty_r), reduceOnly="true")

    # TP1 (MARK_PRICE, reduceOnly)
    tp_cfg = cfg.get("take_profit", {}) or {}
    partial_pct1 = float(tp_cfg.get("partial_pct", partial_pct))
    qty_tp1 = max(step, _round_step(qty_r * (partial_pct1/100.0), step))
    tp1_r = _round_tick(tp1, tick)
    try:
        _place_order_with_retries(client, symbol=symbol, side=side_sl, type="TAKE_PROFIT_MARKET",
                                  workingType="MARK_PRICE", stopPrice=str(tp1_r),
                                  quantity=str(qty_tp1), reduceOnly="true")
    except Exception:
        pass

    # TP2 (valgfri): beregn via R = |entry - stop|
    tp2_rtn = None
    partial2_pct = float(tp_cfg.get("partial2_pct", 0.0) or 0.0)
    partial2_r = float(tp_cfg.get("partial2_r", 0.0) or 0.0)
    if partial2_pct > 0.0 and partial2_r > 0.0:
        rdist = abs(entry - stop)
        tp2 = entry + (partial2_r * rdist) * (1 if side=="BUY" else -1)
        tp2_r = _round_tick(tp2, tick)
        rem_qty = max(step, _round_step(qty_r - qty_tp1, step))
        qty_tp2 = max(step, _round_step(qty_r * (partial2_pct/100.0), step))
        qty_tp2 = min(qty_tp2, rem_qty)
        if qty_tp2 >= step:
            try:
                _place_order_with_retries(client, symbol=symbol, side=side_sl, type="TAKE_PROFIT_MARKET",
                                          workingType="MARK_PRICE", stopPrice=str(tp2_r),
                                          quantity=str(qty_tp2), reduceOnly="true")
                tp2_rtn = tp2_r
            except Exception:
                pass

    return {"qty_r": qty_r, "stop_r": stop_r, "tp_r": tp1_r, "tp2_r": tp2_rtn, "partial_pct1": partial_pct1}

def 
def _position_amt(client: UMFutures, symbol: str) -> float:
    try:
        acc = client.account()
        for p in acc.get("positions", []):
            if p.get("symbol")==symbol:
                return float(p.get("positionAmt") or 0.0)
    except Exception:
        return 0.0
    return 0.0

def _last_price(client: UMFutures, symbol: str) -> float:
    try:
        return float(client.ticker_price(symbol)["price"])
    except Exception:
        return 0.0

def _move_stop_closepos(client: UMFutures, symbol: str, side: str, new_stop: float):
    side_sl = "SELL" if side=="BUY" else "BUY"
    try:
        _place_order_with_retries(client, symbol=symbol, side=side_sl, type="STOP_MARKET",
                         stopPrice=str(new_stop), closePosition=True, reduceOnly="true")
        return True
    except Exception:
        return False

# ---------------- main ----------------

def main():
    load_dotenv()
    cfg = apply_overrides(load_config())

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    client = UMFutures(key=api_key, secret=api_secret)

    notifier = Notifier(cfg.get('telegram', {}).get('token_env', 'TELEGRAM_TOKEN'),
                        cfg.get('telegram', {}).get('chat_id_env', 'TELEGRAM_CHAT_ID'),
                        tz_name=cfg.get('timezone','Europe/Oslo'),
                        style=cfg.get('style','lively'))
    symbols = cfg.get("symbols", ["BTCUSDC","ETHUSDC"])
    paper_mode = not bool(cfg.get('live_trading', False))

    meme_cfg = cfg.get("meme", {})
    meme_syms = set(meme_cfg.get("symbols", []))
    meme_max_conc = int(meme_cfg.get("max_concurrent", 1))
    risk_mult_default = float(meme_cfg.get("risk_mult_default", 0.7))
    lev_cap_default = int(meme_cfg.get("lev_cap_default", 10))
    meme_fund_pos = float(meme_cfg.get("funding_pos", 0.0003))
    meme_fund_neg = float(meme_cfg.get("funding_neg", -0.0003))
    meme_atr_cap = float(meme_cfg.get("atr_price_cap", 0.015))
    meme_per = meme_cfg.get("per_symbol", {})

    if cfg.get('logging', {}).get('telegram', False):
        notifier.start_msg(', '.join([s.replace('USDC','') for s in symbols]), paper=paper_mode)

    # Oppstarts-sjekk
    if cfg.get("startup_checks", {}).get("enabled", True):
        try:
            rep = startup_check(client, symbols,
                                expect_isolated=cfg.get("startup_checks",{}).get("expect_isolated", True),
                                max_leverage=cfg.get("startup_checks",{}).get("max_leverage", cfg.get("leverage",{}).get("max",15)),
                                min_usdc_balance=cfg.get("startup_checks",{}).get("min_usdc_balance", 0.0))
            lines = []
            drift = rep.get("drift_ms")
            if drift is not None and abs(drift) > 1000:
                lines.append(f"⚠️ Klokke drift: {drift:.0f}ms (synkroniser tid om nødvendig)")
            for s in rep.get("symbols", []):
                mt = s.get("marginType"); lev = s.get("leverage")
                if mt is None and lev is None: lines.append(f"{s['symbol']}: (Ingen posisjonsdata)")
                else:
                    ok_m = "OK" if s.get("ok_isolated") else "FEIL"
                    ok_l = "OK" if s.get("ok_leverage") else "FEIL"
                    lines.append(f"{s['symbol']}: Margin {mt} [{ok_m}], Leverage {lev}x [{ok_l}]")
            if cfg.get('logging', {}).get('telegram', False) and lines:
                notifier._send("🔎 Binance-sjekk ved oppstart\n" + "\n".join(lines))
        except Exception as e:
            print(f"[WARN] startup check failed: {e}")

    # Autofix
    if cfg.get("autofix", {}).get("enabled", True):
        want_mt = cfg.get("autofix", {}).get("margin_type", "ISOLATED").upper()
        want_lev = int(cfg.get("autofix", {}).get("leverage", cfg.get("leverage", {}).get("base", 12)))
        if paper_mode:
            if cfg.get('logging', {}).get('telegram', False):
                notifier._send(f"🧪 Auto-fix DRY-RUN (paper): {want_mt}/{want_lev}x planlagt på {', '.join(symbols)} (settes i live).")
        else:
            lines = [f"🛠️ Auto-fix (live): mål {want_mt}/{want_lev}x"]
            for sym in symbols:
                try:
                    
            
            
            # --- Anti-pyramidering + re-entry cooldown ---

            # --- Supertrend-trail (kjør tidlig, gjør ingenting hvis posisjon ikke er åpen) ---
            try:
                stcfg = (cfg.get("trail", {}) or {}).get("supertrend", {}) or {}
                if stcfg.get("enabled", True):
                    # er posisjon åpen?
                    entry_px, pos_side = _position_entry_side(client, sym)
                    if entry_px is not None and pos_side in ("LONG","SHORT"):
                        # df for entry-TF (bruk eksisterende df1h_closed hvis mulig)
                        _df = df1h_closed if "df1h_closed" in locals() and df1h_closed is not None else None
                        if _df is None or len(_df) < 30:
                            _df = get_klines(client, sym, cfg["timeframes"]["entry"], limit=150)

                        from indicators import supertrend_from_df
                        st_p = int(stcfg.get("period", 10)); st_m = float(stcfg.get("multiplier", 3.0))
                        st_line, st_dir = supertrend_from_df(_df, st_p, st_m)

                        atr_p = max(5, int(cfg.get("volatility", {}).get("atr_period", 14)))
                        atr_val = _atr_value(_df, atr_p) or 0.0
                        buf = float(stcfg.get("buffer_atr_mult", 0.20)) * atr_val

                        e, cur_sp = _entry_and_stop(client, sym)
                        if e is None:
                            e = float(_df["close"].iloc[-2])  # fallback
                        last_close = float(_df["close"].iloc[-1])

                        # baseline R0 (første gang vi ser posisjonen)
                        R0 = _trail_get_R0(sym)
                        if R0 is None and cur_sp is not None:
                            R0 = abs(float(e) - float(cur_sp))
                            if R0 and R0 > 0:
                                _trail_set_R0(sym, R0)

                        minR = float(stcfg.get("min_gain_R", 0.5))
                        gainR = 999.0
                        if R0 and R0 > 0:
                            if pos_side == "LONG":
                                gainR = (last_close - float(e)) / R0
                            else:
                                gainR = (float(e) - last_close) / R0

                        if (st_line is not None) and (gainR >= minR):
                            if pos_side == "LONG":
                                cand = float(st_line) - buf
                                best = min(cand, last_close*0.999)
                                _tighten_stop_if_better(client, cfg, sym, "LONG", best)
                            else:
                                cand = float(st_line) + buf
                                best = max(cand, last_close*1.001)
                                _tighten_stop_if_better(client, cfg, sym, "SHORT", best)
            except Exception:
                pass

            # --- Binance leverage/margin guard + riktig margin-asset / free balance ---
            try:
                bguard = cfg.get("binance_guard", {}) or {}
                if bguard.get("enabled", True):
                    _ensure_leverage_margin(client, sym, int(bguard.get("leverage", 12)), str(bguard.get("marginType", "ISOLATED")).upper())
                    m_asset = _symbol_margin_asset(client, sym)  # f.eks. USDT eller USDC
                    bals = _futures_balances(client)
                    free_amt = float(bals.get(m_asset, 0.0))

                    # grovt anslag på nødvendig margin krever qty – vi gjør en enkel sanity: må ha noe fri margin
                    if free_amt <= 0:
                        try:
                            print(f"[SKIP] {sym}: Margin asset {m_asset} free=0.0 — sett inn {m_asset} eller bytt kontrakt (ex: BTCUSDT vs BTCUSDC).")
                        except Exception:
                            pass
                        try:
                            df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                            mark_emitted(sym, _closed_slice(df1h_tmp))
                        except Exception:
                            pass
                        continue
            except Exception:
                pass
            # Finn om posisjon er åpen NÅ
            _is_open_now = False
            try:
                if _position_amt(client, sym) != 0.0 or sym in live_positions:
                    _is_open_now = True
            except Exception:
                pass

            # Dersom den VAR åpen forrige runde men er LUKKET nå -> start cooldown
            try:
                if _HAD_OPEN_LAST.get(sym, False) and not _is_open_now:
                    _cooldown_start(sym, _cooldown_hours(cfg))
            except Exception:
                pass

            # Oppdater status for denne runden (før ev. continue)
            _HAD_OPEN_LAST[sym] = _is_open_now

            # Hvis posisjon er åpen -> aldri ta ny entry (one-way mode vern)
            if _is_open_now:
                try:
                    # markér bar som behandlet for å unngå spam i samme bar
                    df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                    mark_emitted(sym, _closed_slice(df1h_tmp))
                except Exception:
                    pass
                continue

            # Hvis cooldown aktiv -> ikke ta entry enda
            if _cooldown_active(sym):
                try:
                    df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                    mark_emitted(sym, _closed_slice(df1h_tmp))
                except Exception:
                    pass
                continue
# --- Anti-pyramidering + re-entry cooldown ---
            # Finn om posisjon er åpen NÅ
            _is_open_now = False
            try:
                if _position_amt(client, sym) != 0.0 or sym in live_positions:
                    _is_open_now = True
            except Exception:
                pass

            # Dersom den VAR åpen forrige runde men er LUKKET nå -> start cooldown
            try:
                if _HAD_OPEN_LAST.get(sym, False) and not _is_open_now:
                    _cooldown_start(sym, _cooldown_hours(cfg))
            except Exception:
                pass

            # Oppdater status for denne runden (før ev. continue)
            _HAD_OPEN_LAST[sym] = _is_open_now

            # Hvis posisjon er åpen -> aldri ta ny entry (one-way mode vern)
            if _is_open_now:
                try:
                    # markér bar som behandlet for å unngå spam i samme bar
                    df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                    mark_emitted(sym, _closed_slice(df1h_tmp))
                except Exception:
                    pass
                continue

            # Hvis cooldown aktiv -> ikke ta entry enda
            if _cooldown_active(sym):
                try:
                    df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                    mark_emitted(sym, _closed_slice(df1h_tmp))
                except Exception:
                    pass
                continue
# Unngå å legge til posisjon hvis en allerede er åpen (one-way mode vern)
            try:
                if _position_amt(client, sym) != 0.0 or sym in live_positions:
                    try:
                        df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                        mark_emitted(sym, _closed_slice(df1h_tmp))
                    except Exception:
                        pass
                    continue
            except Exception:
                pass
res = autofix_symbol(client, sym, desired_margin=want_mt, desired_leverage=want_lev, recvWindow=5000)
                    actions_txt = ", ".join(res.get("actions", []))
                    after_margin = res.get("after", {}).get("margin")
                    after_lev = res.get("after", {}).get("leverage")
                    lines.append(f"{sym}: {actions_txt} → Etter: {after_margin}/{after_lev}x")
                except Exception as e:
                    lines.append(f"{sym}: auto-fix feilet: {e}")
            if cfg.get('logging', {}).get('telegram', False):
                notifier._send("\n".join(lines))

    broker = PaperBroker(cfg.get('fees', {}), notifier=notifier, cfg=cfg)
    live_positions = {}  # symbol -> dict(...)

    ctrl = TeleControl()
    paused = False
    last_tele_time = {}
    pending_pullback = {}
    pending_breakout = {}   # [NY] regime-avhengig breakout (confirm/retest)

    # --- auto-pause state (daglig sperre / tap på rad) ---
    pause_until_ts = 0
    daily_guard_tripped = False

    # --- helpers for porteføljesperrer ---
    from typing import Optional
    def _open_symbols_live() -> set:
        return set(live_positions.keys())

    def _max_concurrent_block(cfg_local: dict) -> Optional[str]:
        cap = int(cfg_local.get("portfolio", {}).get("max_concurrent", 2))
        total = len(live_positions) + len(pending_pullback) + len(pending_breakout)
        if total >= cap:
            return f"porteføljegrense nådd: {total}/{cap}"
        return None

    def _cluster_block(symbol: str, cfg_local: dict) -> Optional[str]:
        clusters = cfg_local.get("portfolio", {}).get("clusters", {})
        mates = set(clusters.get(symbol, []))
        if not mates:
            return None
        busy = _open_symbols_live() | set(pending_pullback.keys()) | set(pending_breakout.keys())
        hit = sorted(mates & busy)
        if hit:
            return f"korrelasjonsvern: {symbol} i klynge med {', '.join(hit)}"
        return None

    def _is_meme(sym: str) -> bool:
        return sym in meme_syms

    def _meme_limits_ok(sym: str) -> bool:
        if not _is_meme(sym): return True
        active_meme = sum(1 for s in live_positions if s in meme_syms)
        return active_meme < meme_max_conc

    while True:
        # Telegram
        for c in ctrl.poll():
            cmd = c["cmd"]; args = c["args"]; chat = c["chat_id"]
            if cmd in ("/help","/commands"):
                notifier._send(
                    "📋 Kommandoer:\n"
                    "/status – dagsstatistikk (live)\n"
                    "/riskmode fixed <USDC>\n"
                    "/riskmode percent <prosent>\n"
                    "/force <SYMBOL> <long|short> (live)\n"
                    "/pause, /resume, /unpause\n", chat_id=chat
                )
            elif cmd == "/pause":
                paused = True; notifier._send("⏸️ Pause aktivert.", chat_id=chat)
            elif cmd in ("/resume", "/unpause"):
                paused = False; pause_until_ts = 0; daily_guard_tripped = False
                notifier._send("▶️ Gjenopptatt.", chat_id=chat)
            elif cmd == "/status" and hasattr(broker, "daily_report_hype"):
                date_str, wins, losses, winrate, pnl_usdc, avg_r, n = broker.daily_report_hype(cfg.get("timezone","Europe/Oslo"))
                notifier._send(f"📅 I dag: {wins}️⃣ vinn / {losses}️⃣ tap  |  Winrate: {winrate:.1f}%  |  PnL: {pnl_usdc:+.2f} USDC", chat_id=chat)
            elif cmd.startswith("/riskmode"):
                if len(args) < 2:
                    notifier._send("Bruk: /riskmode fixed 10  |  /riskmode percent 0.5", chat_id=chat)
                else:
                    mode, val = args[0].lower(), args[1]
                    try:
                        if mode == "fixed":
                            cfg["risk"]["mode"] = "fixed_usdc"; cfg["risk"]["base_usdc"] = float(val); shown = f"{cfg['risk']['base_usdc']} USDC"
                        else:
                            pct = float(val)/100.0 if float(val) > 1 else float(val)
                            cfg["risk"]["mode"] = "percent"; cfg["risk"]["pct"] = pct; shown = f"{pct*100:.2f}%"
                        try:
                            from runtime_overrides import load_overrides, save_overrides
                            ov = load_overrides() or {}; ov["risk"] = cfg["risk"]; save_overrides(ov)
                        except Exception: pass
                        notifier._send(f"✅ Risiko oppdatert: mode={cfg['risk']['mode']} value={shown}", chat_id=chat)
                    except Exception as e:
                        notifier._send(f"⚠️ Kunne ikke oppdatere risiko: {e}", chat_id=chat)

        # Auto-pause vindu aktivt?
        if pause_until_ts and time.time() < pause_until_ts:
            time.sleep(2); continue

        if paused:
            time.sleep(2); continue

        # Daily loss stop / consecutive losses auto-pause
        if not daily_guard_tripped:
            try:
                tz = cfg.get("timezone","Europe/Oslo")
                _date, _wins, _losses, _wr, _pnl_usdc, _avg_r, _n = broker.daily_report_hype(tz)
                risk_cfg = cfg.get("risk", {})

                cons_cap = int(risk_cfg.get("consecutive_losses_pause", 0) or 0)
                pause_hours = int(risk_cfg.get("pause_hours", 0) or 0)
                if cons_cap and _losses >= cons_cap and pause_hours > 0:
                    pause_until_ts = int(time.time()) + pause_hours * 3600
                    daily_guard_tripped = True
                    if cfg.get('logging', {}).get('telegram', False):
                        notifier._send(f"🛑 Auto-pause: {_losses} tap i dag. Pauser nye entries i {pause_hours}t.")

                dll_pct = float(risk_cfg.get("daily_loss_limit_pct", 0.0) or 0.0)
                if dll_pct > 0.0:
                    bal_now = current_balance(paper_mode, broker, client) or 1000.0
                    regime_for_day = _LAST_REGIME_NAME or "trend"
                    limit_usdc = _daily_loss_limit_usd(bal_now, cfg, regime_for_day)

                    if _pnl_usdc <= limit_usdc:
                        hours = max(1, pause_hours)
                        pause_until_ts = int(time.time()) + hours * 3600
                        daily_guard_tripped = True
                        if cfg.get('logging', {}).get('telegram', False):
                            notifier._send(
                                f"🛑 Auto-pause: Dagens PnL {int(_pnl_usdc):+} USDC ≤ limit {int(limit_usdc)}. "
                                f"Pauser {hours}t."
                            )
            except Exception:
                pass

        # --- [NY] PENDING BREAKOUT check (confirm / retest) ---
        for sym, pend in list(pending_breakout.items()):
            try:
                df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=3)
                df1h_tmp_closed = _closed_slice(df1h_tmp)
                if df1h_tmp_closed is None or len(df1h_tmp_closed) < 2:
                    continue
                cur_bar = _bar_id(df1h_tmp_closed)
                expired = cur_bar > pend.get("expires_bar_id", 0)

                mode = pend["mode"]
                side = pend["side"]
                level = float(pend["level"])
                plan = pend["plan"]
                risk_usdc = float(pend["risk_usdc"])
                eff_cfg_snap = pend.get("eff_cfg_snapshot", cfg)
                partial_pct = eff_cfg_snap.get("take_profit",{}).get("partial_pct",40)

                last_close = float(df1h_tmp_closed["close"].iloc[-1])

                should_fill = False
                if mode == "confirm":
                    if side == "BUY":
                        should_fill = (last_close > level)
                    else:
                        should_fill = (last_close < level)
                else:
                    if side == "BUY":
                        should_fill = (last_close <= level)
                    else:
                        should_fill = (last_close >= level)

                if should_fill:
                    if cfg.get('logging', {}).get('telegram', False):
                        if mode == "confirm":
                            txt = f"✅ {sym} BREAKOUT bekreftet — {'LONG' if side=='BUY' else 'SHORT'} (nivå {level:.4f})."
                        else:
                            txt = f"✅ {sym} BREAKOUT→RETEST fylt @ {level:.4f} — {'LONG' if side=='BUY' else 'SHORT'}."
                        notifier._send(txt)

                    if paper_mode:
                        notifier.plan_msg(sym, side, plan['qty'], plan['entry'], plan['stop'], plan['tp_partial'],
                                          risk_usdc, plan['atr'], eff_lev=plan.get('eff_lev', 0.0),
                                          target_lev=plan.get('target_lev', None), note=pend.get("note"))
                        fill_price = level if mode == "retest" else plan['entry']
                        broker.open_trade(sym, side, plan['qty'], fill_price, plan['stop'], plan['tp_partial'],
                                          plan['r_distance'], plan['trail_atr_mult'], risk_usdc)
                    else:
                        try:
                            try:
                                ok, msg = safe_set_symbol_leverage(client, sym, plan.get('target_lev', 12), recvWindow=5000)
                                if not ok and cfg.get('logging', {}).get('telegram', False):
                                    notifier._send(f"ℹ️ {sym}: {msg}")
                            except Exception:
                                pass
                            br = _place_live_bracket(client, cfg, broker, paper_mode, notifier, sym, side, plan['qty'], level, plan['stop'], plan['tp_partial'], partial_pct)
                            live_positions[sym] = {
                                "side": side, "qty": br["qty_r"], "entry": level, "stop": br["stop_r"],
                                "tp1": br["tp_r"], "partial_pct": eff_cfg_snap.get("take_profit",{}).get("partial_pct",40),
                                "breakeven_done": False, "trail_mult": plan["trail_atr_mult"]
                            }
                        except Exception as e:
                            if cfg.get('logging', {}).get('telegram', False):
                                notifier._send(f"⚠️ {sym}: klarte ikke å fylle pending breakout: {e}")

                    pending_breakout.pop(sym, None)
                    continue

                if expired:
                    if cfg.get('logging', {}).get('telegram', False):
                        notifier._send(f"⌛ {sym} — BREAKOUT {mode} utløpt uten fill/bekreftelse.")
                    pending_breakout.pop(sym, None)

            except Exception as e:
                print(f"[PENDING BO ERROR] {sym}: {e}")

        # --- HYBRID pending check ---
        for sym, pend in list(pending_pullback.items()):
            try:
                last = float(client.ticker_price(sym)["price"])
            except Exception:
                last = None
            try:
                df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                cur_bar = _bar_id(_closed_slice(df1h_tmp))
            except Exception:
                cur_bar = None

            expired = (cur_bar is not None and cur_bar > pend.get("expires_bar_id", 0))
            should_fill = False
            if last is not None:
                if pend["side"] == "BUY":
                    should_fill = (last <= pend["limit"])
                else:
                    should_fill = (last >= pend["limit"])

            if should_fill:
                side = pend["side"]
                plan = pend["plan"]
                risk_usdc = pend["risk_usdc"]
                eff_cfg_snap = pend.get("eff_cfg_snapshot", cfg)
                partial_pct = eff_cfg_snap.get("take_profit",{}).get("partial_pct",40)

                if cfg.get('logging', {}).get('telegram', False):
                    txt_side = "LONG" if side=="BUY" else "SHORT"
                    notifier._send(f"✅ {sym} {txt_side} — Hybrid pullback fylt @ {last:.2f} (plan {pend['limit']:.2f}).")

                if paper_mode:
                    notifier.plan_msg(sym, side, plan['qty'], plan['entry'], plan['stop'], plan['tp_partial'],
                                      risk_usdc, plan['atr'], eff_lev=plan.get('eff_lev', 0.0),
                                      target_lev=plan.get('target_lev', None), note=pend.get("note"))
                    broker.open_trade(sym, side, plan['qty'], last, plan['stop'], plan['tp_partial'],
                                      plan['r_distance'], plan['trail_atr_mult'], risk_usdc)
                else:
                    try:
                        try:
                            ok, msg = safe_set_symbol_leverage(client, sym, plan.get('target_lev', 12), recvWindow=5000)
                            if not ok and cfg.get('logging', {}).get('telegram', False):
                                notifier._send(f"ℹ️ {sym}: {msg}")
                        except Exception:
                            pass
                        br = _place_live_bracket(client, cfg, broker, paper_mode, notifier, sym, side, plan['qty'], last, plan['stop'], plan['tp_partial'], partial_pct)
                        live_positions[sym] = {
                            "side": side, "qty": br["qty_r"], "entry": last, "stop": br["stop_r"],
                            "tp1": br["tp_r"], "partial_pct": eff_cfg_snap.get("take_profit",{}).get("partial_pct",40),
                            "breakeven_done": False, "trail_mult": plan["trail_atr_mult"]
                        }
                    except Exception as e:
                        if cfg.get('logging', {}).get('telegram', False):
                            notifier._send(f"⚠️ {sym}: klarte ikke å fylle hybrid-ordre: {e}")

                pending_pullback.pop(sym, None)
                continue

            if expired:
                if cfg.get('logging', {}).get('telegram', False):
                    notifier._send(f"⌛ {sym} — Hybrid pullback utløpt uten fill. Avbryter.")
                pending_pullback.pop(sym, None)

        for sym in symbols:
            try:
                df4h = get_klines(client, sym, cfg["timeframes"]["trend"], limit=600)
                df1h = get_klines(client, sym, cfg["timeframes"]["entry"], limit=1000)
                df1h_closed = _closed_slice(df1h)
                if df1h_closed is None or len(df1h_closed) < 30:
                    continue

                mscore, regime = compute_market_score(df4h, df1h_closed)
                regime_name = _regime_with_hysteresis(mscore)
                eff_cfg = apply_regime_overrides(deepcopy(cfg), regime_name)

                # >>> Anti-spam TIDLIG
                if not should_emit(
                    sym,
                    df1h_closed,
                    cooldown_bars=eff_cfg.get("portfolio", {}).get("cooldown_bars", 2),
                    throttle_seconds=eff_cfg.get("telegram", {}).get("throttle_seconds", 60),
                ):
                    continue

                # (A) Adaptiv ATR/price guard
                skip_vol, why_vol, atrp, limitp = volatility_guard(df1h_closed, regime, eff_cfg)
                if skip_vol:
                    mark_emitted(sym, df1h_closed)
                    if cfg.get('logging', {}).get('telegram', False):
                        notifier._send(f"⚠️ {sym}: Hopper over – {why_vol}")
                    continue

                bias = compute_trend_bias(df4h, eff_cfg)
                if bias == TrendBias.FLAT:
                    mark_emitted(sym, df1h_closed)
                    continue

                # (B) Max samtidige
                blk = _max_concurrent_block(eff_cfg)
                if blk:
                    mark_emitted(sym, df1h_closed)
                    if cfg.get('logging', {}).get('telegram', False):
                        notifier._send(f"🚧 {sym}: {blk}.")
                    continue

                # (C) Korrelasjonsvern
                corr = _cluster_block(sym, eff_cfg)
                if corr:
                    mark_emitted(sym, df1h_closed)
                    if cfg.get('logging', {}).get('telegram', False):
                        notifier._send(f"🧩 {sym}: {corr}. Hopper over.")
                    continue

                pb = pullback_signal(df1h_closed, bias, eff_cfg)
                bo = breakout_signal(df1h_closed, bias, eff_cfg)

                # Konfluens-skalering: trades tas, men sizing justeres etter PB/BO
                sz = eff_cfg.get("entries", {}).get("sizing", {}) or {}
                both_mult = float(sz.get("both_signals_mult", 1.0))
                single_mult = float(sz.get("single_signal_mult", 0.75))
                signal_mult = both_mult if (pb and bo) else single_mult

                if not pb and not bo:
                    # --- Mean Reversion i RANGE-regime ---
                    if regime == Regime.RANGE and bool(eff_cfg.get("entries",{}).get("mean_reversion",{}).get("enabled", True)):
                        mr_dir = mean_reversion_signal(df1h_closed, eff_cfg)  # "BUY" | "SELL" | None
                        if mr_dir:
                            side = mr_dir  # "BUY" eller "SELL"
                            # Fundingfilter gjelder fortsatt
                            try: fr = get_funding_rate(client, sym) if eff_cfg.get("funding_filter", {}).get("enabled", True) else 0.0
                            except Exception: fr = 0.0
                            if eff_cfg.get("funding_filter", {}).get("enabled", True):
                                if side == "BUY" and fr >= eff_cfg["funding_filter"]["threshold_pos"]:
                                    mark_emitted(sym, df1h_closed); continue
                                if side == "SELL" and fr <= eff_cfg["funding_filter"]["threshold_neg"]:
                                    mark_emitted(sym, df1h_closed); continue

                            # Plan bygges med bias fra MR-retningen
                            plan = build_order_plan(df1h_closed, TrendBias.LONG if side=='BUY' else TrendBias.SHORT, eff_cfg)
                            if plan is None:
                                mark_emitted(sym, df1h_closed); continue

                            # Mean reversion tweaks (raskere TP, lavere lev, halv risiko)
                            mrc = eff_cfg.get("entries",{}).get("mean_reversion",{})
                            risk_mult = float(mrc.get("risk_mult", 0.5))
                            tp_r = float(mrc.get("tp_r", 1.2))
                            trail_mult = float(mrc.get("trail_mult", 1.6))
                            lev_cap_mr = int(mrc.get("lev_cap", 10))

                            bal = current_balance(paper_mode, broker, client) or 1000.0
                            r\n                # --- Volatilitets-vern (ATR%%) ---
try:
    vcfg = cfg.get("volatility", {}) or {}
    if vcfg.get("enabled", True):
        per = int(vcfg.get("atr_period", 14))
        atrp = _atr_percent(df1h_closed, per)
        if atrp is not None:
            scale_thr = float(vcfg.get("atrp_scale_threshold", 0.015))
            block_thr = float(vcfg.get("atrp_block_threshold", 0.035))
            if atrp >= block_thr:
                # blokkér entry nå (for denne bar)
                try:
                    df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                    mark_emitted(sym, _closed_slice(df1h_tmp))
                except Exception:
                    pass
                try:
                    print(f"[SKIP] {sym}: ATR%%={atrp:.3%} >= {block_thr:.3%} (block).")
                except Exception:
                    pass
                continue
            elif atrp >= scale_thr:
                mult = float(vcfg.get("atrp_scale", 0.85))
                risk_usdc *= max(0.1, min(2.0, mult))
                try:
                    print(f"[INFO] {sym}: ATR%%={atrp:.3%} >= {scale_thr:.3%} (scale x{mult}).")
                except Exception:
                    pass
except Exception:
    pass\nisk_usdc = choose_risk_usdc(bal, eff_cfg, {"win_rate": 0.5, "avg_rr": 2.0}) * risk_mult
                # --- Myke filtre: skaler risiko etter VWAP / Supertrend / OBV ---
                try:
                    filter_mult = 1.0
                    filt = cfg.get("filters", {}) or {}
                    last_close = float(df1h_closed["close"].iloc[-1])
                    bias_dir = str(bias).upper() if bias is not None else "FLAT"
                
                    # VWAP (session)
                    vwcfg = (filt.get("vwap") or {})
                    if vwcfg.get("enabled", True):
                        from indicators import vwap_session_from_df
                        vwap_val = vwap_session_from_df(df1h_closed)
                        if vwap_val is not None:
                            sc = float(vwcfg.get("scale", 0.85))
                            if bias_dir == "LONG" and last_close < vwap_val:
                                filter_mult *= sc
                            elif bias_dir == "SHORT" and last_close > vwap_val:
                                filter_mult *= sc
                
                    # Supertrend
                    stcfg = (filt.get("supertrend") or {})
                    if stcfg.get("enabled", True):
                        from indicators import supertrend_from_df
                        st_p = int(stcfg.get("period", 10)); st_m = float(stcfg.get("multiplier", 3.0))
                        st_line, st_dir = supertrend_from_df(df1h_closed, st_p, st_m)
                        sc = float(stcfg.get("scale", 0.80))
                        if st_dir in (-1, 1):
                            if bias_dir == "LONG" and st_dir < 0:
                                filter_mult *= sc
                            elif bias_dir == "SHORT" and st_dir > 0:
                                filter_mult *= sc
                
                    # OBV (valgfri)
                    obvcfg = (filt.get("obv") or {})
                    if obvcfg.get("enabled", False):
                        from indicators import obv_from_df
                        lb = int(obvcfg.get("lookback", 50))
                        sc = float(obvcfg.get("scale", 0.80))
                        obv = obv_from_df(df1h_closed)
                        if len(obv) > lb + 2:
                            import numpy as _np
                            y = _np.array(obv[-lb:])
                            x = _np.arange(lb)
                            slope_obv = _np.polyfit(x, y, 1)[0]
                            y_p = _np.array(df1h_closed["close"].iloc[-lb:])
                            slope_price = _np.polyfit(x, y_p, 1)[0]
                            # enkel divergensregel: pris opp men OBV ikke -> skaler; motsatt for short
                            if (bias_dir == "LONG" and slope_price > 0 and slope_obv <= 0) or                (bias_dir == "SHORT" and slope_price < 0 and slope_obv >= 0):
                                filter_mult *= sc
                
                    risk_usdc *= max(0.1, min(2.0, filter_mult))
                except Exception as _e:
                    # ikke fall over ende ved små problemer
                    pass

                risk_usdc *= max(0.1, min(2.0, signal_mult))
                            plan["tp_partial"] = plan["entry"] + (tp_r * plan["r_distance"]) * (1 if side=='BUY' else -1)
                            plan["trail_atr_mult"] = min(plan["trail_atr_mult"], trail_mult)

                            qty = position_qty(plan["entry"], plan["stop"], risk_usdc)
                            if qty <= 0:
                                mark_emitted(sym, df1h_closed); continue

                            conf = 0.5  # nøytral confidence i range for MR
                            target_lev = min(map_confidence_to_leverage(conf, eff_cfg), lev_cap_mr)
                            eff_lev = (plan['entry'] * qty) / max(bal, 1e-9)

                            # Send signal i Telegram
                            now = int(time.time())
                            throttle = int(eff_cfg.get("telegram", {}).get("throttle_seconds", 60))
                            if now - last_tele_time.get(sym, 0) >= throttle:
                                notifier.signal_msg(sym, 'LONG' if side=='BUY' else 'SHORT', 'mean-reversion', regime=regime, mscore=mscore)
                                last_tele_time[sym] = now

                            if paper_mode:
                                notifier.plan_msg(sym, side, qty, plan['entry'], plan['stop'], plan['tp_partial'], risk_usdc, plan['atr'], eff_lev=eff_lev, target_lev=target_lev, note='MR in RANGE')
                                broker.open_trade(sym, side, qty, plan['entry'], plan['stop'], plan['tp_partial'], plan['r_distance'], plan['trail_atr_mult'], risk_usdc)
                                mark_emitted(sym, df1h_closed)
                            else:
                                try:
                                    ok, msg = safe_set_symbol_leverage(client, sym, target_lev, recvWindow=5000)
                                    if not ok and eff_cfg.get('logging', {}).get('telegram', False): notifier._send(f"ℹ️ {sym}: {msg}")
                                except Exception: pass
                                try:
                                    br = _place_live_bracket(client, eff_cfg, broker, paper_mode, notifier, sym, side, qty, plan['entry'], plan['stop'], plan['tp_partial'], eff_cfg.get('take_profit',{}).get('partial_pct',40))
                                    live_positions[sym] = {"side": side, "qty": br["qty_r"], "entry": plan['entry'], "stop": br["stop_r"], "tp1": br["tp_r"], "partial_pct": eff_cfg.get("take_profit",{}).get("partial_pct",40), "breakeven_done": False, "trail_mult": plan["trail_atr_mult"]}
                                except Exception:
                                    pass
                                mark_emitted(sym, df1h_closed)
                            continue
                    mark_emitted(sym, df1h_closed)
                    continue

                # Funding filter (meme vs normal)
                try: fr = get_funding_rate(client, sym) if eff_cfg.get("funding_filter", {}).get("enabled", True) else 0.0
                except Exception: fr = 0.0
                if _is_meme(sym):
                    if bias == TrendBias.LONG and fr >= meme_fund_pos:
                        mark_emitted(sym, df1h_closed); continue
                    if bias == TrendBias.SHORT and fr <= meme_fund_neg:
                        mark_emitted(sym, df1h_closed); continue
                else:
                    if eff_cfg.get("funding_filter", {}).get("enabled", True):
                        if bias == TrendBias.LONG and fr >= eff_cfg["funding_filter"]["threshold_pos"]:
                            mark_emitted(sym, df1h_closed); continue
                        if bias == TrendBias.SHORT and fr <= eff_cfg["funding_filter"]["threshold_neg"]:
                            mark_emitted(sym, df1h_closed); continue

                plan = build_order_plan(df1h_closed, bias, eff_cfg)
                if plan is None:
                    mark_emitted(sym, df1h_closed); continue

                # Meme ATR/price extra-cap
                if _is_meme(sym):
                    close_last = float(df1h_closed["close"].iloc[-1])
                    atr_last = float(plan["atr"])
                    if close_last > 0 and (atr_last / close_last) > meme_atr_cap:
                        mark_emitted(sym, df1h_closed)
                        if cfg.get('logging', {}).get('telegram', False):
                            notifier._send(f"⚠️ {sym}: Hopper over (ATR/price {atr_last/close_last:.3%} > {meme_atr_cap:.1%})")
                        continue

                conf_raw = compute_confidence(df4h, df1h_closed, bias, pb, bo)
                conf = max(0.0, min(1.0, 0.5*min(mscore/100.0,1.0) + 0.5*conf_raw))
                target_lev = map_confidence_to_leverage(conf, eff_cfg)

                if _is_meme(sym):
                    lev_cap = int(meme_per.get(sym, {}).get("lev_cap", lev_cap_default))
                    target_lev = min(target_lev, lev_cap)

                bal = current_balance(paper_mode, broker, client) or 1000.0
                r\n                # --- Funding-aware risiko (skaler/blokker) ---
try:
    fcfg = cfg.get("funding", {}) or {}
    if fcfg.get("enabled", True):
        fr = _premium_funding_rate(client, sym)  # per 8t
        abs_fr, penal = _dir_penalized(bias, fr, bool(fcfg.get("directional", True)))
        sc_thr = float(fcfg.get("scale_threshold", 0.0005))
        bl_thr = float(fcfg.get("block_threshold", 0.0012))
        if penal and abs_fr >= bl_thr:
            # blokkér entry
            try:
                df1h_tmp = get_klines(client, sym, cfg["timeframes"]["entry"], limit=2)
                mark_emitted(sym, _closed_slice(df1h_tmp))
            except Exception:
                pass
            try:
                print(f"[SKIP] {sym}: Funding {fr:.4%} (penal) >= block {bl_thr:.4%}.")
            except Exception:
                pass
            continue
        elif penal and abs_fr >= sc_thr:
            mult = float(fcfg.get("scale", 0.85))
            risk_usdc *= max(0.1, min(2.0, mult))
            try:
                print(f"[INFO] {sym}: Funding {fr:.4%} (penal) >= scale {sc_thr:.4%} -> risk x{mult}.")
            except Exception:
                pass
except Exception:
    pass\nisk_usdc = choose_risk_usdc(bal, eff_cfg, {"win_rate": 0.5, "avg_rr": 2.0})
                if _is_meme(sym):
                    rm = float(meme_per.get(sym, {}).get("risk_mult", risk_mult_default))
                    risk_usdc *= rm

                note = None
                if regime == Regime.RANGE:
                    risk_usdc *= 0.5
                    plan["tp_partial"] = plan["entry"] + (1.0 * plan["r_distance"]) * (1 if bias == TrendBias.LONG else -1)
                    plan["trail_atr_mult"] = max(1.2, plan["trail_atr_mult"] * 0.7)
                    target_lev = min(target_lev, 10)
                    note = "Range-modus: halverer risiko, lavere lev, tar profitt raskere."
                elif regime == Regime.STRONG_TREND:
                    plan["tp_partial"] = plan["entry"] + (1.8 * plan["r_distance"]) * (1 if bias == TrendBias.LONG else -1)
                    plan["trail_atr_mult"] = max(plan["trail_atr_mult"], 2.8)
                    note = "Sterk trend: lar gevinst løpe med litt romsligere trailing."

                qty = position_qty(plan["entry"], plan["stop"], risk_usdc)
                if qty <= 0:
                    mark_emitted(sym, df1h_closed); continue

                # HYBRID auto
                ep = cfg.get("entry_policy", {})
                mode = (ep.get("mode") or "instant").lower()
                pb_price_mode = (ep.get("pullback", {}).get("price_mode") or "fib_mid").lower()
                ema_dist_frac = float(ep.get("pullback", {}).get("ema_distance_atr_frac", 0.25))
                expiry_bars_cfg = ep.get("pullback", {}).get("expiry_bars", {"range":2,"trend":2,"strong":1})
                min_conf = float(ep.get("pullback", {}).get("min_confidence", 0.70))

                # --- [NY] breakout policy config ---
                bo_cfg = ep.get("breakout", {})
                confirm_bars = int(bo_cfg.get("confirm_bars", 1))
                retest_expire_bars = int(bo_cfg.get("retest_expire_bars", 2))

                use_hybrid_pullback = False
                if pb and not bo:
                    if mode == "hybrid":
                        use_hybrid_pullback = True
                    elif mode == "auto":
                        ema20 = _ema20_last(df1h_closed)
                        far_from_ema = False
                        if ema20 is not None:
                            far_from_ema = abs(plan["entry"] - ema20) >= ema_dist_frac * plan["atr"]
                        regime_ok = (regime in (Regime.TREND, Regime.STRONG_TREND))
                        if regime_ok and (far_from_ema or conf >= min_conf):
                            use_hybrid_pullback = True

                if _is_meme(sym) and not _meme_limits_ok(sym):
                    mark_emitted(sym, df1h_closed)
                    if cfg.get('logging', {}).get('telegram', False):
                        notifier._send(f"🚧 {sym}: Maks antall meme-posisjoner ({meme_max_conc}) nådd. Hopper over.")
                    continue

                # --- Regime-avhengig håndtering av BREAKOUT ---
                if bo:
                    b = eff_cfg["entries"]["breakout"]
                    lookback = int(b.get("lookback", 10))
                    level = _breakout_level(df1h_closed, bias, lookback)
                    side = "BUY" if bias == TrendBias.LONG else "SELL"

                    if regime == Regime.STRONG_TREND:
                        pass
                    elif regime == Regime.TREND:
                        cur_bar = _bar_id(df1h_closed)
                        expiry = cur_bar + max(1, confirm_bars)

                        plan_snapshot = build_order_plan(df1h_closed, bias, eff_cfg)
                        if plan_snapshot is None:
                            mark_emitted(sym, df1h_closed); continue

                        bal2 = current_balance(paper_mode, broker, client) or 1000.0
                        risk2 = choose_risk_usdc(bal2, eff_cfg, {"win_rate": 0.5, "avg_rr": 2.0})
                        risk2 *= max(0.1, min(2.0, signal_mult))
                        qty2 = position_qty(plan_snapshot["entry"], plan_snapshot["stop"], risk2)
                        if qty2 <= 0:
                            mark_emitted(sym, df1h_closed); continue
                        plan_snapshot["qty"] = qty2
                        plan_snapshot["eff_lev"] = (plan_snapshot["entry"] * qty2) / max(bal2, 1e-9)
                        plan_snapshot["target_lev"] = map_confidence_to_leverage(
                            max(0.0, min(1.0, 0.5*min(mscore/100.0,1.0) + 0.5*compute_confidence(df4h, df1h_closed, bias, pb, bo))),
                            eff_cfg
                        )

                        pending_breakout[sym] = {
                            "mode": "confirm",
                            "side": side,
                            "level": float(level),
                            "expires_bar_id": int(expiry),
                            "plan": plan_snapshot,
                            "risk_usdc": risk2,
                            "eff_cfg_snapshot": eff_cfg,
                            "note": f"Breakout bekreftelse {confirm_bars} bar"
                        }

                        if cfg.get('logging', {}).get('telegram', False):
                            notifier._send(f"⏳ {sym} BREAKOUT — venter {confirm_bars} bar bekreftelse over nivå {level:.4f} ({'LONG' if side=='BUY' else 'SHORT'}).")
                        mark_emitted(sym, df1h_closed)
                        continue

                    else:  # RANGE
                        cur_bar = _bar_id(df1h_closed)
                        expiry = cur_bar + max(1, retest_expire_bars)

                        plan_snapshot = build_order_plan(df1h_closed, bias, eff_cfg)
                        if plan_snapshot is None:
                            mark_emitted(sym, df1h_closed); continue

                        bal2 = current_balance(paper_mode, broker, client) or 1000.0
                        risk2 = choose_risk_usdc(bal2, eff_cfg, {"win_rate": 0.5, "avg_rr": 2.0})
                        risk2 *= max(0.1, min(2.0, signal_mult))
                        qty2 = position_qty(plan_snapshot["entry"], plan_snapshot["stop"], risk2)
                        if qty2 <= 0:
                            mark_emitted(sym, df1h_closed); continue
                        plan_snapshot["qty"] = qty2
                        plan_snapshot["eff_lev"] = (plan_snapshot["entry"] * qty2) / max(bal2, 1e-9)
                        plan_snapshot["target_lev"] = map_confidence_to_leverage(
                            max(0.0, min(1.0, 0.5*min(mscore/100.0,1.0) + 0.5*compute_confidence(df4h, df1h_closed, bias, pb, bo))),
                            eff_cfg
                        )

                        pending_breakout[sym] = {
                            "mode": "retest",
                            "side": side,
                            "level": float(level),
                            "expires_bar_id": int(expiry),
                            "plan": plan_snapshot,
                            "risk_usdc": risk2,
                            "eff_cfg_snapshot": eff_cfg,
                            "note": "Breakout→Retest"
                        }

                        if cfg.get('logging', {}).get('telegram', False):
                            notifier._send(f"🎯 {sym} BREAKOUT→RETEST limit @ {level:.4f} (utløper om {retest_expire_bars} bar{'er' if retest_expire_bars>1 else ''}) — {'LONG' if side=='BUY' else 'SHORT'}.")
                        mark_emitted(sym, df1h_closed)
                        continue

                # HYBRID auto pullback
                use_hybrid_pullback = False
                if pb and not bo:
                    if mode == "hybrid":
                        use_hybrid_pullback = True
                    elif mode == "auto":
                        ema20 = _ema20_last(df1h_closed)
                        far_from_ema = False
                        if ema20 is not None:
                            far_from_ema = abs(plan["entry"] - ema20) >= ema_dist_frac * plan["atr"]
                        regime_ok = (regime in (Regime.TREND, Regime.STRONG_TREND))
                        if regime_ok and (far_from_ema or conf >= min_conf):
                            use_hybrid_pullback = True

                if _is_meme(sym) and not _meme_limits_ok(sym):
                    mark_emitted(sym, df1h_closed)
                    if cfg.get('logging', {}).get('telegram', False):
                        notifier._send(f"🚧 {sym}: Maks antall meme-posisjoner ({meme_max_conc}) nådd. Hopper over.")
                    continue

                side = "BUY" if bias == TrendBias.LONG else "SELL"
                eff_lev = (plan['entry'] * qty) / max(bal, 1e-9)

                if use_hybrid_pullback:
                    if pb_price_mode == "ema20":
                        ema20v = _ema20_last(df1h_closed)
                        limit_price = float(ema20v) if ema20v else _fib_mid_from_entry_stop(side, plan["entry"], plan["stop"])
                    else:
                        limit_price = _fib_mid_from_entry_stop(side, plan["entry"], plan["stop"])

                    if regime == Regime.STRONG_TREND:
                        expiry_bars = int(expiry_bars_cfg.get("strong", 1))
                    elif regime == Regime.RANGE:
                        expiry_bars = int(expiry_bars_cfg.get("range", 2))
                    else:
                        expiry_bars = int(expiry_bars_cfg.get("trend", 2))

                    cur_bar = _bar_id(df1h_closed)
                    expires = cur_bar + expiry_bars

                    plan_local = {
                        "qty": qty,
                        "entry": plan["entry"],
                        "stop": plan["stop"],
                        "tp_partial": plan["tp_partial"],
                        "r_distance": plan["r_distance"],
                        "trail_atr_mult": plan["trail_atr_mult"],
                        "atr": plan["atr"],
                        "eff_lev": eff_lev,
                        "target_lev": target_lev
                    }

                    pending_pullback[sym] = {
                        "side": side,
                        "limit": float(limit_price),
                        "expires_bar_id": int(expires),
                        "plan": plan_local,
                        "note": f"Hybrid pullback ({pb_price_mode})",
                        "regime": regime,
                        "mscore": mscore,
                        "risk_usdc": risk_usdc,
                        "eff_cfg_snapshot": eff_cfg
                    }

                    if cfg.get('logging', {}).get('telegram', False):
                        reasons = []
                        if conf >= min_conf: reasons.append(f"conf {conf:.2f}")
                        ema20v = _ema20_last(df1h_closed)
                        if ema20v is not None:
                            dist = abs(plan["entry"] - ema20v)
                            reasons.append(f"EMAΔ={dist:.2f} (ATR={plan['atr']:.2f})")
                        rtxt = ("; ".join(reasons)) if reasons else "-"
                        notifier._send(
                            f"🎯 {sym} {'LONG' if side=='BUY' else 'SHORT'} pullback → HYBRID limit @ {limit_price:.2f} "
                            f"(utløper om {expiry_bars} bar{'er' if expiry_bars>1 else ''}) | {rtxt}"
                        )
                    mark_emitted(sym, df1h_closed)
                    continue

                # Throttle + signal
                now = int(time.time())
                throttle = int(eff_cfg.get("telegram", {}).get("throttle_seconds", 60))
                if now - last_tele_time.get(sym, 0) >= throttle:
                    reasons = []
                    if pb: reasons.append("pullback")
                    if bo: reasons.append("breakout")
                    notifier.signal_msg(sym, bias, "+".join(reasons) if reasons else "-", regime=regime, mscore=mscore)
                    last_tele_time[sym] = now

                # LIVE / PAPER exec
                if paper_mode:
                    notifier.plan_msg(sym, side, qty, plan['entry'], plan['stop'], plan['tp_partial'],
                                      risk_usdc, plan['atr'], eff_lev=eff_lev, target_lev=target_lev, note=note)
                    broker.open_trade(sym, side, qty, plan['entry'], plan['stop'], plan['tp_partial'],
                                      plan['r_distance'], plan['trail_atr_mult'], risk_usdc)
                    mark_emitted(sym, df1h_closed)
                else:
                    try:
                        ok, msg = safe_set_symbol_leverage(client, sym, target_lev, recvWindow=5000)
                        if not ok and eff_cfg.get('logging', {}).get('telegram', False):
                            notifier._send(f"ℹ️ {sym}: {msg}")
                    except Exception as e:
                        if eff_cfg.get('logging', {}).get('telegram', False):
                            notifier._send(f"ℹ️ {sym}: kunne ikke sette leverage: {e}")

                    notifier.plan_msg(sym, side, qty, plan['entry'], plan['stop'], plan['tp_partial'],
                                      risk_usdc, plan['atr'], eff_lev=eff_lev, target_lev=target_lev, note=note)
                    try:
                        br = _place_live_bracket(client, cfg, broker, paper_mode, notifier, sym, side, qty, plan['entry'], plan['stop'], plan['tp_partial'],
                                                 eff_cfg.get("take_profit",{}).get("partial_pct",40))
                        live_positions[sym] = {
                            "side": side, "qty": br["qty_r"], "entry": plan["entry"], "stop": br["stop_r"],
                            "tp1": br["tp_r"], "partial_pct": eff_cfg.get("take_profit",{}).get("partial_pct",40),
                            "breakeven_done": False, "trail_mult": plan["trail_atr_mult"]
                        }
                    except Exception:
                        mark_emitted(sym, df1h_closed)
                        continue

                    try:
                        lp = _last_price(client, sym)
                        tick, _ = _exchange_filters(client, sym)
                        if lp > 0:
                            if side=="BUY":
                                new_sl = max(live_positions[sym]["stop"], lp - live_positions[sym]["trail_mult"]*plan["atr"])
                                new_sl = _round_tick(new_sl, tick)
                                if new_sl > live_positions[sym]["stop"]:
                                    if _move_stop_closepos(client, sym, side, new_sl):
                                        live_positions[sym]["stop"] = new_sl
                            else:
                                new_sl = min(live_positions[sym]["stop"], lp + live_positions[sym]["trail_mult"]*plan["atr"])
                                new_sl = _round_tick(new_sl, tick)
                                if new_sl < live_positions[sym]["stop"]:
                                    if _move_stop_closepos(client, sym, side, new_sl):
                                        live_positions[sym]["stop"] = new_sl
                    except Exception:
                        pass

                    mark_emitted(sym, df1h_closed)

            except Exception as e:
                print(f"[ERROR] {sym}: {e}")

        # LIVE trailing & BE
        if not paper_mode and live_positions:
            for sym, o in list(live_positions.items()):
                try:
                    amt = abs(_position_amt(client, sym))
                    if amt <= 1e-12:
                        live_positions.pop(sym, None)
                        continue
                    if not o["breakeven_done"]:
                        orig_qty = o["qty"]
                        if amt <= max(1e-9, orig_qty*(1.0 - o["partial_pct"]/100.0) + 1e-9):
                            be = o["entry"]
                            tick, _ = _exchange_filters(client, sym)
                            be_r = _round_tick(be, tick)
                            if _move_stop_closepos(client, sym, o["side"], be_r):
                                o["breakeven_done"] = True
                    lp = _last_price(client, sym)
                    if lp > 0:
                        tick, _ = _exchange_filters(client, sym)
                        if o["side"]=="BUY":
                            candidate = lp - o["trail_mult"] * abs(o["entry"] - o["stop"])
                            candidate = _round_tick(candidate, tick)
                            if candidate > o["stop"]:
                                if _move_stop_closepos(client, sym, o["side"], candidate):
                                    o["stop"] = candidate
                        else:
                            candidate = lp + o["trail_mult"] * abs(o["entry"] - o["stop"])
                            candidate = _round_tick(candidate, tick)
                            if candidate < o["stop"]:
                                if _move_stop_closepos(client, sym, o["side"], candidate):
                                    o["stop"] = candidate
                except Exception as e:
                    print(f"[LIVE sync WARN] {sym}: {e}")

        # =========================
        # 15m Micro Entries (EMA21/55 + StochRSI)
        # =========================
        try:
            from strategy import micro_signal_ema_stoch, should_emit_micro, mark_emitted_micro, ema as _ema
        except Exception as e:
            print("[15m init error] " + str(e))
            micro_signal_ema_stoch = None
            should_emit_micro = None
            mark_emitted_micro = None

        if micro_signal_ema_stoch is not None and should_emit_micro is not None:
            try:
                micro_tf = cfg.get("timeframes", {}).get("micro", "15m")
                micro = cfg.get("micro_entry", {})                
                if "PENDING_MICRO" not in globals():
                    global PENDING_MICRO
                    PENDING_MICRO = {}
                smart_cfg = micro.get("smart_entry", {"enabled": True, "method": "pullback_to_ema", "max_wait_bars": 2, "limit_offset_atr": 0.25, "ema_ref":"fast"})
                for sym in symbols:
                    try:
                        df4h_m = get_klines(client, sym, cfg["timeframes"]["trend"], limit=400)
                        df1h_m = get_klines(client, sym, cfg["timeframes"]["entry"], limit=400)
                        df1h_m_closed = _closed_slice(df1h_m)
                        df15 = get_klines(client, sym, micro_tf, limit=400)
                        df15_closed = _closed_slice(df15)
                        # Kill-switch: block entries on extreme ATR spikes
                        from strategy import _atr_median, KILL_UNTIL
                        ks_cfg = cfg.get("killswitch", {"enabled": True, "atr_mult": 3.0, "cooloff_min": 15})
                        if ks_cfg.get("enabled", True):
                            cur_atr, med_atr = _atr_median(df15_closed, period=cfg.get("stops",{}).get("atr_period",14), lookback=200)
                            if cur_atr and med_atr and cur_atr > ks_cfg.get("atr_mult",3.0) * med_atr:
                                KILL_UNTIL[sym] = int(time.time()) + int(60*ks_cfg.get("cooloff_min",15))
                                notifier.warn_throttled("killswitch", f"{sym}: ⛔ Killswitch — ATR spike ({cur_atr:.2f} > {ks_cfg.get('atr_mult',3.0)}× median {med_atr:.2f}). Pause nye entries i {ks_cfg.get('cooloff_min',15)} min.")
                                mark_emitted_micro(sym, df15_closed)
                                continue
                            if KILL_UNTIL.get(sym, 0) > int(time.time()):
                                left = KILL_UNTIL[sym] - int(time.time())
                                notifier.info_throttled("killswitch", f"{sym}: killswitch aktiv ({left}s igjen).", min_seconds=60)
                                mark_emitted_micro(sym, df15_closed)
                                continue

                        if df1h_m_closed is None or df15_closed is None:
                            continue

                        # Handle pending smart-entry først
                        if sym in PENDING_MICRO:
                            pend = PENDING_MICRO[sym]
                            bar_ms = df15_closed["close_time"].iloc[-1] - df15_closed["close_time"].iloc[-2]
                            if df15_closed["close_time"].iloc[-1] >= pend["deadline_bar_ts"]:
                                try:
                                    if paper_mode:
                                        side_bin = "BUY" if pend["side"]=="long" else "SELL"
                                        notifier.plan_msg(sym, side_bin, pend['qty'], pend['plan']['entry'], pend['plan']['stop'], pend['plan']['tp_partial'],
                                                          pend['risk'], pend['plan']['atr'], eff_lev=(pend['plan']['entry']*pend['qty'])/max(1.0,pend['risk']),
                                                          target_lev=10, note="15m smart-timeout -> MARKET")
                                    else:
                                        side_bin = "BUY" if pend["side"]=="long" else "SELL"
                                        br = _place_live_bracket(client, cfg, broker, paper_mode, notifier, sym, side_bin, pend['qty'], pend['plan']['entry'], pend['plan']['stop'], pend['plan']['tp_partial'],
                                                                 cfg.get('take_profit',{}).get('partial_pct',40))
                                        live_positions[sym] = {"side": side_bin, "qty": br["qty_r"], "entry": pend['plan']['entry'],
                                                               "stop": br["stop_r"], "tp1": br["tp_r"], "partial_pct": cfg.get("take_profit",{}).get("partial_pct",40),
                                                               "breakeven_done": False, "trail_mult": pend['plan']["trail_atr_mult"]}
                                except Exception as e:
                                    try:
                                        notifier._send("⚠️ " + str(sym) + ": live bracket feilet (timeout->market): " + str(e))
                                    except Exception:
                                        print("[15m timeout live err] " + str(sym) + ": " + str(e))
                                finally:
                                    PENDING_MICRO.pop(sym, None)
                            else:
                                price_now = float(df15_closed["close"].iloc[-1])
                                efast = _ema(list(df15_closed["close"]), micro.get("ema_fast",21))[-1]
                                eslow = _ema(list(df15_closed["close"]), micro.get("ema_slow",55))[-1]
                                ema_ref = efast if smart_cfg.get("ema_ref","fast")=="fast" else eslow
                                target = pend["target_price"]
                                ready = (price_now <= max(target, ema_ref)) if pend["side"]=="long" else (price_now >= min(target, ema_ref))
                                if ready:
                                    side_bin = "BUY" if pend["side"]=="long" else "SELL"
                                    try:
                                        if paper_mode:
                                            notifier.plan_msg(sym, side_bin, pend['qty'], pend['plan']['entry'], pend['plan']['stop'], pend['plan']['tp_partial'],
                                                              pend['risk'], pend['plan']['atr'], eff_lev=(pend['plan']['entry']*pend['qty'])/max(1.0,pend['risk']),
                                                              target_lev=10, note="15m smart-entry filled")
                                            broker.open_trade(sym, side_bin, pend['qty'], pend['plan']['entry'], pend['plan']['stop'], pend['plan']['tp_partial'],
                                                              pend['plan']['r_distance'], pend['plan']['trail_atr_mult'], pend['risk'])
                                        else:
                                            br = _place_live_bracket(client, cfg, broker, paper_mode, notifier, sym, side_bin, pend['qty'], pend['plan']['entry'], pend['plan']['stop'], pend['plan']['tp_partial'],
                                                                     cfg.get('take_profit',{}).get('partial_pct',40))
                                            live_positions[sym] = {"side": side_bin, "qty": br["qty_r"], "entry": pend['plan']['entry'],
                                                                   "stop": br["stop_r"], "tp1": br["tp_r"], "partial_pct": cfg.get("take_profit",{}).get("partial_pct",40),
                                                                   "breakeven_done": False, "trail_mult": pend['plan']["trail_atr_mult"]}
                                    except Exception as e:
                                        try:
                                            notifier._send("⚠️ " + str(sym) + ": live bracket feilet (smart-treff): " + str(e))
                                        except Exception:
                                            print("[15m smart live err] " + str(sym) + ": " + str(e))
                                    finally:
                                        PENDING_MICRO.pop(sym, None)
                            continue

                        # Trend bias fra 4H
                        bias_m = compute_trend_bias(df4h_m, cfg)
                        if str(bias_m) == "TrendBias.FLAT" or bias_m is None:
                            continue

                        # Gating
                        if not should_emit_micro(sym, df15_closed, cooldown_bars=cfg.get("portfolio",{}).get("cooldown_bars",2),
                                                 throttle_seconds=cfg.get("telegram",{}).get("throttle_seconds",60)):
                            continue

                        # Signal på 15m
                        side_sig = micro_signal_ema_stoch(df15_closed, bias_m, cfg)
                        if not side_sig:
                            continue

                        try:
                            if hasattr(notifier, "micro_debug"):
                                notifier.micro_debug(sym, side_sig, tf="15m")
                            else:
                                notifier._send(("🟢 " if side_sig=='long' else "🔴 ") + str(sym) + " micro-entry (15m)")
                        except Exception:
                            print("[15m signal] " + str(sym) + " " + str(side_sig))

                        # Preflight balance
                        bal_m = current_balance(paper_mode, broker, client) or 0.0
                        if bal_m <= 0.0:
                            try:
                                notifier._send(str(sym) + " 15m " + str(side_sig).upper() + " blokkert: ingen USDC på konto")
                            except Exception:
                                print("[15m block] no USDC for " + str(sym))
                            mark_emitted_micro(sym, df15_closed)
                            continue

                        # Funding filter
                        try:
                            fr = get_funding_rate(client, sym) if cfg.get("funding_filter", {}).get("enabled", True) else 0.0
                        except Exception:
                            fr = 0.0
                        if cfg.get("funding_filter", {}).get("enabled", True):
                            if side_sig == "long" and fr >= cfg["funding_filter"]["threshold_pos"]:
                                try:
                                    notifier._send(str(sym) + " 15m LONG blokkert av funding (" + str(fr) + ")")
                                except Exception:
                                    pass
                                mark_emitted_micro(sym, df15_closed); continue
                            if side_sig == "short" and fr <= cfg["funding_filter"]["threshold_neg"]:
                                try:
                                    notifier._send(str(sym) + " 15m SHORT blokkert av funding (" + str(fr) + ")")
                                except Exception:
                                    pass
                                mark_emitted_micro(sym, df15_closed); continue

                        # Bygg plan
                        plan_m = build_order_plan(df15_closed, TrendBias.LONG if side_sig=='long' else TrendBias.SHORT, cfg)
                        if plan_m is None:
                            mark_emitted_micro(sym, df15_closed); continue

                        # Risk sizing/lev
                        mscore_m, regime_m = compute_market_score(df4h_m, df1h_m_closed)
                        conf_raw_m = compute_confidence(df4h_m, df1h_m_closed, TrendBias.LONG if side_sig=='long' else TrendBias.SHORT, True, False)
                        conf_m = max(0.0, min(1.0, 0.5*min(mscore_m/100.0,1.0) + 0.5*conf_raw_m))
                        target_lev_m = map_confidence_to_leverage(conf_m, cfg)
                        risk_usdc_m = choose_risk_usdc(bal_m, cfg, {"win_rate": 0.5, "avg_rr": 2.0})
                        qty_m = position_qty(plan_m["entry"], plan_m["stop"], risk_usdc_m)

                        limits = cfg.get("portfolio",{}).get("limits",{})
                        if limits:
                            longs = sum(1 for p in live_positions.values() if p.get("side","").upper()=="BUY")
                            shorts= sum(1 for p in live_positions.values() if p.get("side","").upper()=="SELL")
                            if side_sig=='long' and longs >= int(limits.get("max_longs", 999)):
                                notifier.warn_throttled("max_longs", f"{sym}: stoppet — maks longs nådd.", limits.get("keys",{}).get("max_longs",120))
                                mark_emitted_micro(sym, df15_closed); continue
                            if side_sig=='short' and shorts >= int(limits.get("max_shorts", 999)):
                                notifier.warn_throttled("max_shorts", f"{sym}: stoppet — maks shorts nådd.", limits.get("keys",{}).get("max_shorts",120))
                                mark_emitted_micro(sym, df15_closed); continue
                            memes = set(limits.get("memecoins", []))
                            def _notional(symb):
                                try:
                                    pos = live_positions.get(symb);
                                    if not pos: return 0.0
                                    qty = float(pos.get("qty",0))
                                    price = float(df15_closed["close"].iloc[-1]) if symb==sym else float(get_klines(client, symb, cfg["timeframes"]["entry"], limit=2)["close"].iloc[-1])
                                    return qty*price
                                except Exception:
                                    return 0.0
                            curr_meme_notional = sum(_notional(s) for s in live_positions.keys() if s in memes)
                            future_add = float(plan_m['entry']*qty_m) if sym in memes else 0.0
                            if curr_meme_notional + future_add > float(limits.get("max_memecoin_notional", 9e9)):
                                notifier.warn_throttled("min_notional", f"{sym}: stoppet — memecoin notional cap nådd.", 180)
                                mark_emitted_micro(sym, df15_closed); continue
                            open_count = len(live_positions)
                            scale_steps = limits.get("risk_scale_steps", [1.0, 0.7, 0.5])
                            idx = max(0, min(len(scale_steps)-1, open_count))
                            scale = float(scale_steps[idx])
                            if scale < 1.0:
                                risk_usdc_m = float(risk_usdc_m) * scale
                                qty_m = position_qty(plan_m["entry"], plan_m["stop"], risk_usdc_m)
                                if qty_m <= 0:
                                    notifier.warn_throttled("no_balance", f"{sym}: stoppet — qty<=0 etter nedskalering.", 120)
                                    mark_emitted_micro(sym, df15_closed); continue
                        if qty_m <= 0:
                            try:
                                notifier._send(str(sym) + " 15m " + str(side_sig).upper() + " blokkert: qty<=0 fra sizing")
                            except Exception:
                                pass
                            mark_emitted_micro(sym, df15_closed); continue

                        # Smart-entry: vent på pullback/EMA
                        if smart_cfg.get("enabled", True):
                            atr = plan_m["atr"]
                            offset = float(smart_cfg.get("limit_offset_atr", 0.25))
                            efast = _ema(list(df15_closed["close"]), micro.get("ema_fast",21))[-1]
                            eslow = _ema(list(df15_closed["close"]), micro.get("ema_slow",55))[-1]
                            ema_ref = efast if smart_cfg.get("ema_ref","fast")=="fast" else eslow
                            if side_sig == "long":
                                target_price = max(plan_m["entry"] - offset*atr, ema_ref)
                            else:
                                target_price = min(plan_m["entry"] + offset*atr, ema_ref)
                            bar_ms = df15_closed["close_time"].iloc[-1] - df15_closed["close_time"].iloc[-2]
                            deadline = int(df15_closed["close_time"].iloc[-1] + smart_cfg.get("max_wait_bars",2)*bar_ms)
                            PENDING_MICRO[sym] = {"side": side_sig, "bar_ts": int(df15_closed["close_time"].iloc[-1]), "deadline_bar_ts": int(deadline),
                                                  "target_price": float(target_price), "plan": plan_m, "qty": float(qty_m), "risk": float(risk_usdc_m)}
                            try:
                                notifier._send(str(sym) + " 15m " + str(side_sig).upper() + ": smart-entry aktiv — mål " + "{:.6f}".format(target_price) + ", timeout " + str(smart_cfg.get("max_wait_bars",2)) + " barer")
                            except Exception:
                                pass
                            mark_emitted_micro(sym, df15_closed)
                            continue

                        side_bin = "BUY" if side_sig == "long" else "SELL"
                        if paper_mode:
                            try:
                                notifier.plan_msg(sym, side_bin, qty_m, plan_m['entry'], plan_m['stop'], plan_m['tp_partial'],
                                                  risk_usdc_m, plan_m['atr'], eff_lev=(plan_m['entry']*qty_m)/max(1.0,bal_m),
                                                  target_lev=target_lev_m, note="15m micro-entry (no smart-wait)")
                            except Exception:
                                pass
                            broker.open_trade(sym, side_bin, qty_m, plan_m['entry'], plan_m['stop'], plan_m['tp_partial'],
                                              plan_m['r_distance'], plan_m['trail_atr_mult'], risk_usdc_m)
                        else:
                            try:
                                br = _place_live_bracket(client, cfg, broker, paper_mode, notifier, sym, side_bin, qty_m, plan_m['entry'], plan_m['stop'], plan_m['tp_partial'],
                                                         cfg.get('take_profit',{}).get('partial_pct',40))
                                live_positions[sym] = {"side": side_bin, "qty": br["qty_r"], "entry": plan_m['entry'],
                                                       "stop": br["stop_r"], "tp1": br["tp_r"], "partial_pct": cfg.get("take_profit",{}).get("partial_pct",40),
                                                       "breakeven_done": False, "trail_mult": plan_m["trail_atr_mult"]}
                            except Exception as e:
                                try:
                                    notifier._send("⚠️ " + str(sym) + ": klarte ikke å fylle 15m entry: " + str(e))
                                except Exception:
                                    print("[15m live err] " + str(sym) + ": " + str(e))
                        mark_emitted_micro(sym, df15_closed)
                    except Exception as e:
                        print("[15m ERROR] " + str(sym) + ": " + str(e))
            except Exception as e:
                print("[15m block init error] " + str(e))

        # --- Loop-sleep for å unngå REST-spam / rate-limit ---
        time.sleep(1.5)

if __name__ == "__main__":
    main()
