# Crypto Dream Bot (USDC Perps) — with Telegram

## Install
```
pip install -r requirements.txt
```

## Set credentials (never hardcode in code)
```
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
export TELEGRAM_TOKEN=your_bot_token
export TELEGRAM_CHAT_ID=your_chat_id
```

## Run (paper mode)
```
python main.py
```
You should receive Telegram messages for signals & plans if env vars are set.
