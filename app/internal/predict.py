import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .utils import yfinance_config, model_prediction

def predict_model(ticker) -> float:
    yfinance_config['ticker'] = ticker
    yfinance_prediction = model_prediction(yfinance_config['ticker'], yfinance_config['period'])
    return yfinance_prediction.predict()