import streamlit as st
import numpy as np
import pandas as pd
from preprocessamento import coletar_dados_btc, preparar_dados_lstm
from tensorflow.keras.models import load_model
import os

st.title("Previsão de Alta ou Baixa - BTC com LSTM e Fear & Greed Index")

# Seleção do horizonte de previsão
horizonte = st.selectbox("Selecione horizonte de previsão:", options=[1, 6, 12, 24], format_func=lambda x: f"{x} horas")

# Carregar dados e preparar
@st.cache_data(show_spinner=True)
def carregar_dados(h):
    df = coletar_dados_btc(periodo="90d", intervalo="1h")
    X, y, scaler = preparar_dados_lstm(df, janela=24, horizonte=h, incluir_sentimento=True)
    return X, y, scaler

X, y, scaler = carregar_dados(horizonte)

# Carregar modelo (se existir)
modelo_path = "modelo_lstm.h5"
if os.path.exists(modelo_path):
    model = load_model(modelo_path)
else:
    st.warning("Modelo não encontrado! Treine o modelo antes de usar.")
    st.stop()

# Botão para prever
if st.button("Prever tendência"):
    latest_seq = X[-1:]
    pred_prob = model.predict(latest_seq)[0][0]
    pred_class = int(pred_prob > 0.5)
    st.write(f"Probabilidade de alta: {pred_prob:.2f}")
    if pred_class == 1:
        st.success("O modelo prevê: ALTA")
    else:
        st.error("O modelo prevê: BAIXA")

    st.line_chart(scaler.inverse_transform(X[-1][:,0].reshape(-1,1)), height=200, use_container_width=True)