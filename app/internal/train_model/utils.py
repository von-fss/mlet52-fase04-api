from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from ..utils import read_yfinance

class LSTM_model_train:
    def __init__(self, ticker: str, period: str):
        self.data : pd.DataFrame = read_yfinance(ticker, period)
        self.ticker = ticker
        self.X = []
        self.y = []
        self.model = Sequential()

    def _create_dataset(self, scaled_data, time_step=1) -> np:
        X, y = [], []
        for i in range(len(scaled_data) - time_step - 1):
            X.append(scaled_data[i:(i + time_step), 0])
            y.append(scaled_data[i + time_step, 0])
        return np.array(X), np.array(y)

    def _LSTM_parameters(self) -> None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)
        time_step = 3
        self.X, self.y = self._create_dataset(scaled_data, time_step)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)

    def _LSTMModel(self) -> None:
        ############# Update neural layers for next steps (v-0.1) ##########
        # model = Sequential()
        self.model.add(Input(shape=(self.X.shape[1], 1)))
        self.model.add(LSTM(units=50, activation='relu', return_sequences=False))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        ####################################################################
        # return model

    def LSTMModel_train(self) -> None:
        self._LSTM_parameters()
        self._LSTMModel()
        self.model.fit(self.X, self.y, epochs=20, batch_size=15)

    def save(self) -> None:
        self.model.save(r'app\\internal\\train_model\\models\\model_{}.keras'.format(self.ticker))