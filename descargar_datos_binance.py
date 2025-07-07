# descargar_datos_binance.py

import os
from dotenv import load_dotenv
from binance.client import Client
import pandas as pd
import time
from datetime import datetime, timedelta

# -----------------------
# CONFIGURACIÓN
# -----------------------
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR  # puedes cambiarlo a 1DAY, etc.
start_str = "1 Jan, 2023"  # fecha de inicio
output_csv = f"{symbol}_{interval}_historial.csv"

# -----------------------
# CONEXIÓN BINANCE
# -----------------------
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# -----------------------
# FUNCIÓN PARA DESCARGAR
# -----------------------
def descargar_velas_en_rango(client, symbol, interval, start_str):
    print(f"\n🔄 Descargando datos desde {start_str}...")
    df = pd.DataFrame(client.get_historical_klines(symbol, interval, start_str))
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    print(f"✅ Descargadas {len(df)} velas de {symbol} ({interval})")
    return df

# -----------------------
# DESCARGAR Y GUARDAR
# -----------------------
df = descargar_velas_en_rango(client, symbol, interval, start_str)
df.to_csv(output_csv)
print(f"💾 Datos guardados en: {output_csv}")
