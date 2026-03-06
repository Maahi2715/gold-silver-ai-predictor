import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("AI Gold & Silver Price Predictor")

# ---------------- FETCH DATA ----------------

gold = yf.download("GC=F", period="6mo")
silver = yf.download("SI=F", period="6mo")
usd_inr = yf.download("INR=X", period="1d")

gold_close = gold["Close"].squeeze()
silver_close = silver["Close"].squeeze()

gold_price_usd_ounce = float(gold_close.iloc[-1])
silver_price_usd_ounce = float(silver_close.iloc[-1])
usd_to_inr = float(usd_inr["Close"].iloc[-1])

# ounce → gram
gold_usd_gram = gold_price_usd_ounce / 31.1035
silver_usd_gram = silver_price_usd_ounce / 31.1035

# convert to rupees
gold_inr_gram = gold_usd_gram * usd_to_inr
silver_inr_gram = silver_usd_gram * usd_to_inr

# ---------------- DISPLAY TODAY PRICE ----------------

st.header("Today's Price")

col1, col2 = st.columns(2)

with col1:
    st.metric("Gold (USD / gram)", round(gold_usd_gram,2))
    st.metric("Gold (₹ / gram)", round(gold_inr_gram,2))

with col2:
    st.metric("Silver (USD / gram)", round(silver_usd_gram,2))
    st.metric("Silver (₹ / gram)", round(silver_inr_gram,2))

# ---------------- AI MODEL ----------------

gold_df = gold.reset_index()
silver_df = silver.reset_index()

gold_df["Day"] = np.arange(len(gold_df))
silver_df["Day"] = np.arange(len(silver_df))

gold_model = LinearRegression()
silver_model = LinearRegression()

gold_model.fit(gold_df[["Day"]], gold_df["Close"])
silver_model.fit(silver_df[["Day"]], silver_df["Close"])

next_day = np.array([[len(gold_df)+1]])

gold_pred = gold_model.predict(next_day).item()
silver_pred = silver_model.predict(next_day).item()

# ---------------- PREDICTION ----------------

# ---------------- PREDICTION ----------------

st.header("Tomorrow Prediction")

# USD prediction (ounce)
gold_pred_usd_ounce = gold_model.predict(next_day).item()
silver_pred_usd_ounce = silver_model.predict(next_day).item()

# convert ounce → gram
gold_pred_usd_gram = gold_pred_usd_ounce / 31.1035
silver_pred_usd_gram = silver_pred_usd_ounce / 31.1035

# convert to rupees
gold_pred_inr_gram = gold_pred_usd_gram * usd_to_inr
silver_pred_inr_gram = silver_pred_usd_gram * usd_to_inr


col1, col2 = st.columns(2)

with col1:
    st.write("Gold Tomorrow (USD / gram):", round(gold_pred_usd_gram,2))
    st.write("Gold Tomorrow (₹ / gram):", round(gold_pred_inr_gram,2))

with col2:
    st.write("Silver Tomorrow (USD / gram):", round(silver_pred_usd_gram,2))
    st.write("Silver Tomorrow (₹ / gram):", round(silver_pred_inr_gram,2))

# ---------------- CHART ----------------

chart_data = pd.DataFrame({
    "Gold": gold_close.values.flatten(),
    "Silver": silver_close.values.flatten()
})

st.header("Price Trend")

st.line_chart(chart_data)