Crypto DREAMBOT – Steg 1 (Stabilitetspatch)

Hva som er endret (høynivå):
1) data_feed.py
   - Beholder nå 'close_time' som INT (millisekunder) både som index og kolonne.
   - Sikrer konsistent bar-ID/anti-spam og samsvar med ws_manager.

2) strategy.py
   - Ryddet anti-spam: enkel og entydig bruk av 'close_time' (ms).
   - Separat 15m anti-spam med egne funksjonsnavn (ingen skygge/konflikt).
   - Implementert KILL_UNTIL + _atr_median for 15m killswitch.
   - Micro EMA/StochRSI beholdt, men gjort robust mot manglende data.

3) main.py
   - Fikset _place_order_with_retries til å kalle Binance-klienten (ikke seg selv).
   - _place_live_bracket tar nå (client, cfg, broker, paper_mode, notifier, ...) og kalles konsekvent.
   - _last_price ryddet (ingen dødkode).
   - Konsistent bruk av bar-ID og mark_emitted på alle skip-paths.
   - Tydelig loop-sleep (1.5s) for å unngå REST-spam.
   - Små justeringer i meldinger og feilhåndtering.

4) notifier.py
   - Robust Notifier med throttling (info/warn/error) + micro_debug.
   - Bruker env-variabler (token/chat-id) oppgitt i config.json som ENV-NAVN.
   - Degraderer til stdout hvis token/chat mangler.

Slik oppdaterer du:
- Pakk ut og erstatt filene i prosjektmappen din (backup anbefales).
- Kjør først i PAPER (config.json: "live_trading": false) og verifiser Telegram-varsler og signalflyt.
- Når alt ser bra ut, kan du slå på LIVE igjen.

Tips:
- Om du bruker WebSocket: sett "ws.enabled": true i config.json for lavere REST-bruk.
- Gi meg logg/feilmeldinger i Telegram/terminal hvis noe ser rart ut, så tar vi Steg 2 (lønnsomhetstiltak).
