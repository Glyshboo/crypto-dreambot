# --- Optional WebSocket helper (safe) ---
ws_mgr = None
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


def _place_order_with_retries(client, **kwargs):
    """
    Robust wrapper som faktisk kaller Binance-klienten (ikke seg selv).
    Brukes for MARKET/STOP_MARKET/TAKE_PROFIT_MARKET ordrer.
    """
    delay = 0.2
    last_exc = None
    for attempt in range(4):
        try:
            return client.new_order(**kwargs)
        except Exception as e:
            last_exc = e
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    if last_exc:
        raise last_exc

def _place_live_bracket(client: UMFutures, cfg: dict, broker, paper_mode: bool, notifier, symbol: str, side: str, qty: float, entry: float, stop: float, tp1: float, partial_pct: float):
    # Filters & rounding
    try:
        flt = get_symbol_filters(client, symbol)
        tick = flt['tick']; step = flt['step']
    except Exception:
        tick, step = _exchange_filters(client, symbol)  # fallback
        flt = {'min_qty': step, 'min_notional': 0.0}

    # Cap against margin & normalize against minQty/minNotional
    try:
        bal = current_balance(paper_mode, broker, client)
    except Exception:
        bal = 0.0
    lev = cfg.get('leverage', {}).get('base', 12)
    maxq = max_qty_from_margin(bal or 0.0, lev, entry, 0.97)
    qty = min(qty, maxq)
    _, qty_norm = normalize_price_qty(client, symbol, entry, qty)
    qty_r = max(step, _round_step(qty_norm, step))

    # Market entry
    _place_order_with_retries(client, symbol=symbol, side=side, type="MARKET", quantity=str(qty_r))

    # Protective stop (MARK_PRICE)
    side_sl = "SELL" if side=="BUY" else "BUY"
    stop_r = _round_tick(stop, tick)
    try:
        _place_order_with_retries(client, symbol=symbol, side=side_sl, type="STOP_MARKET", workingType="MARK_PRICE",
                         stopPrice=str(stop_r), closePosition=True)
    except Exception:
        _place_order_with_retries(client, symbol=symbol, side=side_sl, type="STOP_MARKET", workingType="MARK_PRICE",
                         stopPrice=str(stop_r), quantity=str(qty_r), reduceOnly="true")

    # Partial TP (MARK_PRICE)
    qty_tp = max(step, _round_step(qty_r * (partial_pct/100.0), step))
    tp_r = _round_tick(tp1, tick)
    try:
        _place_order_with_retries(client, symbol=symbol, side=side_sl, type="TAKE_PROFIT_MARKET", workingType="MARK_PRICE",
                         stopPrice=str(tp_r), quantity=str(qty_tp), reduceOnly="true")
    except Exception:
        pass
    return {"qty_r": qty_r, "stop_r": stop_r, "tp_r": tp_r}

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
                            risk_usdc = choose_risk_usdc(bal, eff_cfg, {"win_rate": 0.5, "avg_rr": 2.0}) * risk_mult
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
                risk_usdc = choose_risk_usdc(bal, eff_cfg, {"win_rate": 0.5, "avg_rr": 2.0})
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
