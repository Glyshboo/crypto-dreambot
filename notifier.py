import os, time, json, requests

class Notifier:
    def __init__(self, token_env: str, chat_id_env: str, tz_name: str = "Europe/Oslo", style: str = "lively"):
        # token_env/chat_id_env er ENV-VARIABEL-NAVN (ikke selve verdien)
        self.token = os.getenv(token_env, "")
        self.chat_id = os.getenv(chat_id_env, "")
        self.tz_name = tz_name
        self.style = style
        self._last_sent = {}  # key -> last_ts

    # --- intern helper ---
    def _should_send(self, key: str, min_seconds: int) -> bool:
        now = int(time.time())
        last = self._last_sent.get(key, 0)
        if now - last >= max(0, int(min_seconds)):
            self._last_sent[key] = now
            return True
        return False

    def _send(self, text: str, chat_id: str = None):
        chat = chat_id or self.chat_id
        if not self.token or not chat:
            # fallback til stdout
            print("[TELEGRAM DISABLED] " + text)
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
            requests.post(url, json=payload, timeout=8)
        except Exception as e:
            print("[TELEGRAM ERROR] " + str(e) + " | " + text)

    # --- throttle wrappers ---
    def info_throttled(self, key: str, text: str, min_seconds: int = 60):
        if self._should_send(f"info:{key}", min_seconds):
            self._send("ℹ️ " + text)

    def warn_throttled(self, key: str, text: str, min_seconds: int = 60):
        if self._should_send(f"warn:{key}", min_seconds):
            self._send("⚠️ " + text)

    def error_throttled(self, key: str, text: str, min_seconds: int = 60):
        if self._should_send(f"err:{key}", min_seconds):
            self._send("🛑 " + text)

    # --- standard meldinger ---
    def start_msg(self, symbols_txt: str, paper: bool):
        mode = "🧪 PAPER" if paper else "💹 LIVE"
        self._send(f"🤖 Crypto DREAMBOT startet – {mode}\nSymbols: {symbols_txt}")

    def signal_msg(self, symbol: str, bias: str, reason: str, regime: str, mscore: float):
        arrow = "🟢" if str(bias).lower().endswith("long") or str(bias).lower()=="long" else "🔴"
        self._send(f"{arrow} {symbol} {bias.upper()} – {reason} | Regime: {regime} (score {mscore:.0f})")

    def plan_msg(self, symbol: str, side: str, qty: float, entry: float, stop: float, tp_partial: float,
                 risk_usdc: float, atr: float, eff_lev: float = 0.0, target_lev: int = None, note: str = None, chat_id=None):
        nl = "\n"
        tlev = f"{target_lev}x" if target_lev is not None else "-"
        note_txt = f"{nl}Note: {note}" if note else ""
        self._send(
            f"🧭 {symbol} PLAN {side} qty={qty:.6f}{nl}Entry {entry:.6f}{nl}SL {stop:.6f}{nl}TP1 {tp_partial:.6f}"
            f"{nl}Risk {risk_usdc:.2f} USDC | ATR {atr:.6f}{nl}Lev eff {eff_lev:.2f}× (target {tlev}){note_txt}",
            chat_id=chat_id
        )

    def exit_msg_hype(self, symbol: str, pnl_usdc: float, pnl_r: float, qty: float, entry: float, exit_price: float,
                      atr: float, is_win: bool, day_wins: int, day_losses: int, day_pnl_usdc: float):
        face = "😎" if is_win else "😬"
        self._send(
            f"{face} {symbol} EXIT  PnL {pnl_usdc:+.2f} USDC  ({pnl_r:+.2f}R)\n"
            f"qty={qty:.6f} entry={entry:.6f} exit={exit_price:.6f} ATR={atr:.6f}\n"
            f"📅 I dag: {day_wins} win / {day_losses} loss  |  PnL {day_pnl_usdc:+.2f} USDC"
        )

    # Debug for 15m
    def micro_debug(self, symbol: str, side: str, tf: str = "15m"):
        self.info_throttled(f"micro:{symbol}", f"{symbol} micro-signal ({tf}) → {side.upper()}", 60)
