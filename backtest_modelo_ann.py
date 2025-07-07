# -----------------------------------------
# backtest_modelo_ann.py
# Simulación interactiva de estrategia ANN
# -----------------------------------------

import os
from dotenv import load_dotenv
from binance.client import Client
import pandas as pd
import numpy as np
import ta
from tensorflow.keras.models import load_model
from joblib import load
from collections import Counter
import matplotlib.pyplot as plt

# --------------------------
# CONFIGURACIÓN INTERACTIVA
# --------------------------
interval_dict = {
    "1": Client.KLINE_INTERVAL_1MINUTE,
    "2": Client.KLINE_INTERVAL_5MINUTE,
    "3": Client.KLINE_INTERVAL_1HOUR,
    "4": Client.KLINE_INTERVAL_4HOUR,
    "5": Client.KLINE_INTERVAL_1DAY,
    "6": Client.KLINE_INTERVAL_1WEEK
}

print("\n📌 Elige el intervalo de velas:")
print("1 - 1 minuto")
print("2 - 5 minutos")
print("3 - 1 hora")
print("4 - 4 horas")
print("5 - 1 día")
print("6 - 1 semana")
opcion = input("👉 Ingresa el número de tu elección: ").strip()
interval = interval_dict.get(opcion, Client.KLINE_INTERVAL_1HOUR)

start_date = input("📅 Ingresa la fecha de inicio (ej: 1 Jan, 2024): ").strip() or "1 Jan, 2024"
lookahead = int(input("🔎 ¿Cuántas velas observar después de comprar? [default=5]: ") or "5")
take_profit = float(input("💰 Porcentaje de take profit (ej: 0.02 para 2%): ") or "0.02")
stop_loss = float(input("🛑 Porcentaje de stop loss (ej: 0.015 para 1.5%): ") or "0.015")
threshold = float(input("🎯 Umbral mínimo de predicción para comprar (ej: 0.5): ") or "0.5")

print(f"\n✅ Intervalo: {interval} | Desde: {start_date} | TP: {take_profit*100:.1f}% | SL: {stop_loss*100:.1f}% | Pred ≥ {threshold}")

# --------------------------
# CONEXIÓN Y DATOS
# --------------------------
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

symbol = "BTCUSDT"
klines = client.get_historical_klines(symbol, interval, start_date)

df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# --------------------------
# INDICADORES TÉCNICOS
# --------------------------
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

# --------------------------
# CARGAR MODELO Y SCALER
# --------------------------
model = load_model("modelo_signal_ann.h5")
scaler = load("scaler_signal_ann.pkl")

features = ['rsi', 'macd', 'macd_signal', 'ema_20', 'sma_50',
            'bb_upper', 'bb_lower', 'bb_width', 'volume']
X = df[features].copy()
X_scaled = scaler.transform(X)
y_pred_proba = model.predict(X_scaled)
y_pred = (y_pred_proba >= threshold).astype(int).flatten()
df['pred'] = y_pred

# --------------------------
# BACKTEST
# --------------------------
results = []
capital = [1000.0]  # curva de capital

for i in range(len(df) - lookahead):
    if df['pred'].iloc[i] == 1:
        entry_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead].values

        max_price = np.max(future_prices)
        min_price = np.min(future_prices)

        if (max_price - entry_price) / entry_price >= take_profit:
            result = 'win'
            capital.append(capital[-1] * (1 + take_profit))
        elif (entry_price - min_price) / entry_price >= stop_loss:
            result = 'loss'
            capital.append(capital[-1] * (1 - stop_loss))
        else:
            final_price = future_prices[-1]
            pnl = (final_price - entry_price) / entry_price
            result = 'neutral' if abs(pnl) < 0.005 else ('win' if pnl > 0 else 'loss')
            capital.append(capital[-1] * (1 + pnl))

        results.append(result)

# --------------------------
# RESULTADOS
# --------------------------
summary = Counter(results)
total = sum(summary.values())
print("\n📊 Resultado del Backtest:")
for k in ['win', 'loss', 'neutral']:
    print(f"{k.title():<8}: {summary.get(k, 0)} ({summary.get(k, 0)/total:.2%})")

final_return = (capital[-1] - capital[0]) / capital[0]
print(f"\n💰 Retorno total: {final_return:.2%}")
print(f"📈 Capital final: ${capital[-1]:.2f}")
print(f"📉 Número de operaciones: {total}")

# --------------------------
# GRÁFICO DE CAPITAL
# --------------------------
plt.figure(figsize=(10, 5))
plt.plot(capital, label="Equity Curve", linewidth=2)
plt.title("📈 Curva de Capital Simulada")
plt.xlabel("Operaciones")
plt.ylabel("Saldo ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
