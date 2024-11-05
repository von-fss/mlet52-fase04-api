import yfinance as yf
import pandas as pd

def get_data() -> dict:
    nvda = yf.Ticker("NVDA").history(period="1mo").reset_index()
    nvda['Date'] = pd.to_datetime(nvda['Date']).dt.strftime('%Y-%m-%d')
    return nvda.to_dict(orient="records")