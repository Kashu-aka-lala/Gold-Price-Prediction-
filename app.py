import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- App title ---
st.set_page_config(page_title="Gold Price Predictor", layout="centered")
st.title("📈 Real-Time Gold Price Prediction")
st.markdown("Using a trained LSTM model to predict next gold price value")

# --- Load the model ---
@st.cache_resource
def load_lstm_model():
    return load_model("gold_price_lstm_model.h5")

model = load_lstm_model()

# --- Load the dataset (historical) ---
@st.cache_data
def load_data():
    df = pd.read_csv("Gold Price (2013-2023).csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
    df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
    return df

df = load_data()

# --- Preprocessing ---
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(df['Price'].values.reshape(-1,1))

# --- Prepare last 60 days input ---
window_size = 60
last_60_days = scaled_prices[-window_size:]
X_input = np.reshape(last_60_days, (1, window_size, 1))

# --- Make prediction ---
predicted_scaled_price = model.predict(X_input)
predicted_price = scaler.inverse_transform(predicted_scaled_price)[0][0]

# --- Display ---
last_date = df['Date'].iloc[-1]
predicted_date = last_date + timedelta(days=1)

st.subheader("📊 Last 7 Days of Gold Prices")
st.line_chart(df.set_index('Date')['Price'].tail(7))

st.subheader("📌 Predicted Price for Next Day")
st.metric(label=f"{predicted_date.strftime('%Y-%m-%d')}", value=f"${predicted_price:,.2f}")

# --- Plot historical + predicted ---
st.subheader("🧠 Predicted vs Historical Trend")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'].tail(60), y=df['Price'].tail(60), mode='lines+markers', name='Historical'))
fig.add_trace(go.Scatter(x=[predicted_date], y=[predicted_price], mode='markers+text', name='Predicted',
                         marker=dict(size=12, color='red'), text=["Next"], textposition='top center'))
fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Gold Price", showlegend=True)
st.plotly_chart(fig, use_container_width=True)
X_train = []
y_train = []