import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .utils import model_prediction

def predict_model(ticker) -> float:
    yfinance_prediction = model_prediction(ticker, '5d')
    return yfinance_prediction.predict()