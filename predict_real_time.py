# -----------------------------------------
# predict_real_time.py
# Clasificador ANN de señales de compra en tiempo real
# en ejecución continua cada 60 minutos
# -----------------------------------------

import os
from dotenv import load_dotenv
import pandas as pd
import ta
import numpy as np
from binance.client import Client
from tensorflow.keras.models import load_model
from joblib import load
from datetime import datetime, timedelta
import pytz
import requests
import time

# ---------------------
# INTERVALO INTERACTIVO (por ahora fijo)
# ---------------------
interval = Client.KLINE_INTERVAL_4HOUR
interval_nombre = "4 hora"

# ---------------------
# CARGAR CREDENCIALES Y MODELO
# ---------------------
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)

model = load_model("modelo_signal_ann.h5")
scaler = load("scaler_signal_ann.pkl")

symbol = "BTCUSDT"
limit = 60  # para calcular indicadores

# ---------------------
# FUNCIÓN DE MENSAJE TELEGRAM
# ---------------------
def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": mensaje}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("❌ Error al enviar mensaje a Telegram:", e)

# ---------------------
# LOOP CONTINUO CADA 15 MINUTOS
# ---------------------
while True:
    try:
        print("\n🔄 Ejecutando predicción...")
        # DESCARGAR ÚLTIMAS VELAS
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # CALCULAR INDICADORES
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        df.dropna(inplace=True)

        # PREDICCIÓN
        features = ['rsi', 'macd', 'macd_signal', 'ema_20', 'sma_50',
                    'bb_upper', 'bb_lower', 'bb_width', 'volume']
        X_live = df[features].iloc[-1:].copy()
        X_scaled = scaler.transform(X_live)
        pred = model.predict(X_scaled)[0][0]

        # RESULTADO
        utc_time = X_live.index[0].to_pydatetime()
        utc_minus_5 = utc_time - timedelta(hours=5)
        precio_actual = df['close'].iloc[-1]

        mensaje = f"📊 {symbol} - Intervalo: {interval_nombre}\n📅 Vela: {utc_time.strftime('%Y-%m-%d %H:%M:%S')} UTC / {utc_minus_5.strftime('%Y-%m-%d %H:%M:%S')} UTC-5\n💰 Precio actual: {precio_actual:.2f} USD\n📈 Probabilidad de compra: {pred:.2%}"
        if pred >= 0.5:
            mensaje += "\n✅ Señal del modelo: COMPRAR 🚀"
        else:
            mensaje += "\n⛔ Señal del modelo: NO comprar"

        print("\n" + mensaje)
        enviar_telegram(mensaje)

    except Exception as e:
        print(f"❌ Error durante ejecución: {e}")

    print("⏳ Esperando 60 minutos para la próxima predicción...")
    time.sleep(3600)  # 60 minutos
