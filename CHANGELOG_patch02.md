# CHANGELOG – Patch 02 (isolerte filer)
Dato: 2025-08-25

Denne patchen inneholder kun endringsfiler for å legges rett inn i eksisterende repo uten å overskrive alt.

## Filer i patchen
- `strategy.py`: Ny/regnskrevet strategi med regime-filtre, ADX/volum-gating, ATR‑SL/TP, cooldown og 15m mikro-bekreftelse.
- `notifier.py`: Telegram-varsling med anti-spam cooldown og fallback til print.
- `execution_setup.py`: Klargjøring av symbol-metadata; live-innstillinger gjøres i Patch 03.
- `main.py`: Robust oppstart for paper/backtest, enkel løkke, kobling mot Strategy/Notifier/ExecutionSetup.

## Bruddendringer
- Live-handel er fortsatt deaktivert med vilje i Patch 02. (Kommer i Patch 03.)
- Indikatorer implementert internt i `strategy.py` for å unngå avhengighetsfeil.

## Slik oppdaterer du
Kopier filene over dine eksisterende:

```
/your_repo/
  main.py              (erstatt)
  strategy.py          (erstatt)
  notifier.py          (erstatt)
  execution_setup.py   (erstatt)
```

## Konfig (env)
- `SYMBOL` (default: SOLUSDT)
- `MODE`   (paper | backtest)
- `CSV_4H`, `CSV_1H`, `CSV_15M` (valgfritt) – peker til egne OHLCV-CSV-er med kolonnene open,high,low,close,volume
- `LOOP_SEC` (default 60)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (valgfritt)
- `TG_COOLDOWN_SEC` (default 30)

## Hva ble fikset
- Konsistente imports (ingen sirkulære imports).
- Fail-safes i notifier (ingen exception stopper programmet om Telegram er nede/ikke satt).
- Cooldown mot signal-spam: samme tekst under X sekunder blir droppet.
- Strategy: dedup av identisk side per symbol innenfor tidsvindu.
