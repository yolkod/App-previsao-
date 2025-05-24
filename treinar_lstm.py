import numpy as np
from preprocessamento import coletar_dados_btc, preparar_dados_lstm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def treinar_modelo(janela=24, horizonte=1):
    print("Coletando dados BTC...")
    df = coletar_dados_btc(periodo="90d", intervalo="1h")
    print("Preparando dados...")
    X, y, _ = preparar_dados_lstm(df, janela=janela, horizonte=horizonte, incluir_sentimento=True)

    print(f"Shape dos dados: X={X.shape}, y={y.shape}")

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    model.save("modelo_lstm.h5")
    print("Modelo salvo como modelo_lstm.h5")

if __name__ == "__main__":
    treinar_modelo()