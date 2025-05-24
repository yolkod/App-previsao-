import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volume import OnBalanceVolumeIndicator
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler

def coletar_dados_btc(periodo='90d', intervalo='1h'):
    df = yf.download('BTC-USD', period=periodo, interval=intervalo)
    if df.empty:
        raise ValueError("Erro ao coletar dados do BTC.")
    df = df.dropna()
    return df

def coletar_indicadores(df):
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['EMA'] = EMAIndicator(close=df['Close'], window=14).ema_indicator()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd_diff()
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df.dropna(inplace=True)
    return df

def coletar_fear_greed():
    try:
        url = "https://alternative.me/crypto/fear-and-greed-index/"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        valor = soup.find("div", class_="fgi-value").text.strip()
        return int(valor)
    except Exception:
        return 50

def preparar_dados_lstm(df, janela=24, horizonte=1, incluir_sentimento=True):
    fear_greed = coletar_fear_greed()
    if incluir_sentimento:
        df['FearGreed'] = fear_greed

    df = coletar_indicadores(df)

    df['Target'] = (df['Close'].shift(-horizonte) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['Close', 'RSI', 'EMA', 'MACD', 'OBV']
    if incluir_sentimento:
        features.append('FearGreed')

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    sequencias = []
    alvos = []

    for i in range(len(df) - janela - horizonte):
        seq = df[features].iloc[i:i+janela].values
        alvo = df['Target'].iloc[i+janela]
        sequencias.append(seq)
        alvos.append(alvo)

    X = np.array(sequencias)
    y = np.array(alvos)

    return X, y, scaler