import pandas as pd
from typing import Optional
from binance.um_futures import UMFutures

INTERVAL_MAP = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}

def klines_to_df(klines) -> pd.DataFrame:
    """
    Returnerer en DataFrame med kolonnene:
      ['open','high','low','close','volume','close_time']
    hvor 'close_time' er i millisekunder (int) og er **både** index og kolonne.
    Dette matcher også formen fra WS-manageren, slik at anti-spam og bar-ID blir stabile.
    """
    cols = ['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base','taker_buy_quote','ignore']
    df = pd.DataFrame(klines, columns=cols)
    # Sørg for riktige typer
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    # Behold close_time i millisekunder (Binance-klines er allerede ms)
    df['close_time'] = df['close_time'].astype('int64')
    # Sett index til close_time ms for enkel slicing, men behold kolonnen
    df.set_index('close_time', inplace=True)
    # Returner OHLCV + close_time-kolonne
    out = df[['open','high','low','close','volume']].copy()
    out['close_time'] = out.index.astype('int64')
    return out

def get_klines(client: UMFutures, symbol: str, interval: str, limit: int=500) -> pd.DataFrame:
    data = client.klines(symbol=symbol, interval=INTERVAL_MAP[interval], limit=limit)
    return klines_to_df(data)

def get_funding_rate(client: UMFutures, symbol: str) -> float:
    fr = client.funding_rate(symbol=symbol, limit=1)
    if isinstance(fr, list) and len(fr) > 0:
        try:
            return float(fr[0].get('fundingRate', 0.0))
        except Exception:
            return 0.0
    return 0.0
