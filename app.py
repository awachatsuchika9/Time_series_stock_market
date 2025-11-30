import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("Stock Forecast Demo")
ticker = st.text_input("Ticker", "RELIANCE.NS")
start = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))

if st.button("Download & Show"):
    df = yf.download(ticker, start=start, progress=False)
    st.line_chart(df['Close'])
    st.write(df.tail())
