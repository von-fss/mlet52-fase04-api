import pandas as pd
from .utils import LSTM_model_train
from ..utils import yfinance_config

def train_model(ticker: str) -> None:
    yfinance_config['ticker'] = ticker

    #create model
    lstm_model = LSTM_model_train(yfinance_config['ticker'], yfinance_config['period'])

    #train model
    lstm_model.LSTMModel_train()

    #Save model
    lstm_model.save()