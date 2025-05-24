import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import requests
import datetime
import os

st.set_page_config(page_title="Previsão BTC com LSTM", layout="centered")
st.title("Previsão de Alta ou Baixa - BTC com LSTM e Índice de Medo e Ganância")

# Função: obter índice de medo e ganância
def obter_indice_medo_ganancia():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        resposta = requests.get(url)
        valor = int(resposta.json()["data"][0]["value"])
        return valor
    except:
        return 50

# Função: coletar dados e calcular indicadores
def preparar_dados(periodo="60d", intervalo="1h"):
    df = yf.download("BTC-USD", period=periodo, interval=intervalo)
    if df.empty:
        return None

    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["EMA"] = EMAIndicator(close=df["Close"], window=14).ema_indicator()
    df["MACD"] = MACD(close=df["Close"]).macd()
    df["STOCH"] = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"]).stoch()
    df["CCI"] = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"]).cci()
    df["BOLL"] = BollingerBands(close=df["Close"]).bollinger_mavg()
    df["OBV"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df.dropna(inplace=True)
    return df

# Interface
opcao = st.selectbox("Escolha o horizonte da previsão:", ["5 horas", "10 horas", "1 dia"])
passos_previsao = {"5 horas": 5, "10 horas": 10, "1 dia": 24}[opcao]

# Botão de previsão
if st.button("Prever Alta ou Baixa"):
    st.info("Processando previsão...")

    df = preparar_dados()
    if df is None or len(df) < 100:
        st.error("Dados insuficientes para análise.")
    else:
        fng = obter_indice_medo_ganancia()
        df["FNG"] = fng / 100.0

        features = ["Close", "RSI", "EMA", "MACD", "STOCH", "CCI", "BOLL", "OBV", "FNG"]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features])
        X_input = scaled[-60:].reshape(1, 60, len(features))

        if not os.path.exists("modelo_lstm.h5"):
            st.error("Arquivo do modelo 'modelo_lstm.h5' não encontrado. Faça upload no repositório.")
        else:
            model = load_model("modelo_lstm.h5")
            pred = model.predict(X_input)[0][0]
            movimento = "ALTA" if pred > df["Close"].pct_change().mean() else "BAIXA"
            st.success(f"Previsão para os próximos {passos_previsao} períodos ({opcao}): **{movimento}**")
            st.line_chart(df["Close"])
        
