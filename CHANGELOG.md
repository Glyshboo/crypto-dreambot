# CHANGELOG – Patch 01

## Added
- ADX, volum, VWAP og Supertrend filtre i `strategy.py` (togglet via config).
- Dedup / cooldown for entries per symbol/timeframe.
- Mikro‐bekreftelse (15m) for entries (valgfritt via config).
- Ny `.gitignore` for å holde repo rent for runtime/state filer.

## Fixed
- ATR stop‐loss mismatch: `initial_atr_mult` fra config brukes korrekt i beregning.
- Bedre exception handling og logging i `main.py`.
- Risk: konsistent posisjonsstørrelse, fraksjonert Kelly med caps.

## Changed
- Telegram notifier med rate‐limit og mer detaljert PnL info.
- `backtest.py` og `paper.py` utvidet med rapportering: Sharpe ratio, max drawdown, daglig/ukentlig PnL.

## Removed/Deprecated
- Runtime/state filer (`state.json`, `tg_state.json`, `runtime_overrides.json`) fjernet fra repo (ligger nå kun lokalt).
