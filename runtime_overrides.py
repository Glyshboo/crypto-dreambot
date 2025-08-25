
import json, os
PATH = "runtime_overrides.json"

def load_overrides():
    if os.path.exists(PATH):
        try:
            with open(PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_overrides(obj):
    try:
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except Exception:
        pass
