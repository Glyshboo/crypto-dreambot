
import os, time, requests

class TeleControl:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self.offset = None

    def _get_updates(self):
        if not self.enabled:
            return []
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"timeout": 0}
        if self.offset:
            params["offset"] = self.offset
        try:
            r = requests.get(url, params=params, timeout=10)
            js = r.json()
            return js.get("result", [])
        except Exception:
            return []

    def poll(self):
        if not self.enabled:
            time.sleep(1)
            return []
        out = []
        for upd in self._get_updates():
            self.offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message")
            if not msg: 
                continue
            if str(msg.get("chat", {}).get("id", "")) != str(self.chat_id):
                continue
            text = (msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue
            parts = text.split()
            cmd = parts[0].lower()
            args = parts[1:]
            out.append({"cmd": cmd, "args": args, "chat_id": str(self.chat_id)})
        return out
