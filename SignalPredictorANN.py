# -----------------------------------------------
# SignalPredictorANN.py
# Clasificador binario con ANN usando datos históricos locales
# -----------------------------------------------

import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from joblib import dump

# --------------------------
# 1. CARGAR DATOS DESDE CSV
# --------------------------
ruta_csv = "BTCUSDT_1h_historial.csv"  # Ajusta el nombre si es diferente
df = pd.read_csv(ruta_csv, index_col='timestamp', parse_dates=True)
df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

print(f"✅ Datos cargados: {len(df)} velas")

# --------------------------
# 2. INDICADORES TÉCNICOS
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
# 3. ETIQUETADO (TARGET)
# --------------------------
lookahead = 5
threshold = 0.02  # 2%

df['future_max'] = df['close'].shift(-lookahead).rolling(window=lookahead).max()
df['future_return'] = (df['future_max'] - df['close']) / df['close']
df['target'] = (df['future_return'] >= threshold).astype(int)
df.dropna(inplace=True)

# --------------------------
# 4. SELECCIÓN Y ESCALADO DE VARIABLES
# --------------------------
features = ['rsi', 'macd', 'macd_signal', 'ema_20', 'sma_50',
            'bb_upper', 'bb_lower', 'bb_width', 'volume']

X = df[features].copy()
y = df['target'].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# 5. MODELO ANN CON CLASS_WEIGHT
# --------------------------
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Ponderación para mejorar recall en clase "comprar"
pesos_clase = {0: 1.0, 1: 5.0}

model.fit(X_train, y_train,
          epochs=30,
          batch_size=32,
          validation_data=(X_test, y_test),
          class_weight=pesos_clase,
          verbose=1)

# --------------------------
# 6. EVALUACIÓN
# --------------------------
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)
labels = ['No Comprar (0)', 'Comprar (1)']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción del Modelo")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

print("\n📋 Reporte de Clasificación:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# --------------------------
# 7. GUARDAR MODELO Y SCALER
# --------------------------
model.save("modelo_signal_ann.h5")
dump(scaler, "scaler_signal_ann.pkl")
print("✅ Modelo y scaler guardados.")
