
from typing import Dict, Any

def kelly_position_size(win_rate: float, rr: float) -> float:
    k = win_rate - (1 - win_rate) / max(rr, 1e-6)
    return max(0.0, min(1.0, k))

def choose_risk_usdc(balance: float, cfg: Dict[str, Any], rolling_perf: Dict[str, float]) -> float:
    rcfg = cfg.get("risk", {})
    mode = rcfg.get("mode", "fixed_usdc")
    base = float(rcfg.get("base_usdc", 10))
    pct = float(rcfg.get("pct", 0.005))
    min_r = float(rcfg.get("min_usdc", 5))
    max_r = float(rcfg.get("max_usdc", 50))
    cap_pct = rcfg.get("max_balance_pct", 0.75)
    cap_pct = (cap_pct / 100.0) if cap_pct > 1 else float(cap_pct)

    wr = rolling_perf.get("win_rate", 0.5)
    rr = rolling_perf.get("avg_rr", 2.0)
    k = rcfg.get("kelly_k", 0.3) * kelly_position_size(wr, rr)

    if mode == "percent":
        dyn = balance * pct * (1 + k)
    else:
        dyn = base * (1 + k)

    dyn = max(min_r, min(max_r, dyn))
    dyn = min(dyn, balance * cap_pct)
    return float(dyn)

def position_qty(entry_price: float, stop_price: float, risk_usdc: float) -> float:
    r = abs(entry_price - stop_price)
    if r <= 0:
        return 0.0
    qty = risk_usdc / r
    return max(0.0, qty)
