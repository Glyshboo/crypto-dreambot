# -*- coding: utf-8 -*-
"""
notifier.py — Patch 02
- Telegram varsling med enkel rate-limit/cooldown.
- Faller tilbake til print/log hvis ikke konfigurert.
"""
import os
import time
import threading
from typing import Optional

try:
    import requests
except Exception:
    requests = None

class Notifier:
    def __init__(self, cooldown_sec: int = 30):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.cooldown_sec = cooldown_sec
        self._last_sent_text = ""
        self._last_sent_ts = 0.0
        self._lock = threading.Lock()

    def _can_send(self, text: str) -> bool:
        now = time.time()
        if text == self._last_sent_text and (now - self._last_sent_ts) < self.cooldown_sec:
            return False
        return True

    def send(self, text: str) -> None:
        with self._lock:
            if not self._can_send(text):
                return
            self._last_sent_text = text
            self._last_sent_ts = time.time()
        if self.token and self.chat_id and requests is not None:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
            try:
                requests.post(url, json=payload, timeout=10)
            except Exception as e:
                print(f"[notifier] telegram feilet: {e}; fallback -> print\n{text}")
        else:
            print(text)
