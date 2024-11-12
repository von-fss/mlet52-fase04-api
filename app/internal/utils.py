from keras.layers import LSTM, Dense, Input, Dropout
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import pandas as pd
from ..routers.utils import modelConfig
from dataclasses import dataclass

def read_yfinance(ticker: str, period: str) -> pd.DataFrame:
    nvda: pd.DataFrame = yf.Ticker(ticker).history(period)
    return pd.DataFrame(nvda['Close'])

class model_prediction:
    def __init__(self, ticker, period):
        self.model = load_model(r'app\internal\models\model_{}.keras'.format(ticker))
        self.data = read_yfinance(ticker, period)
        self.X = []
        self.time_step = 3

    def predict(self) -> float:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)

        for i in range(len(scaled_data)-self.time_step):
            self.X.append(scaled_data[i:(i+self.time_step), 0])

        self.X = np.array(self.X)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))

        prediction = self.model.predict(self.X)
        prediction = scaler.inverse_transform(prediction)
        return prediction[0]

@dataclass
class expandedModelConfig(modelConfig):
    def add_dropout(self, model, last_layer) -> None:
        if self.dropout and self.nn_layers > 1 and not last_layer:
            model.add(Dropout(self.dropout_value))

def create_lstm_model(X) -> Sequential:
    config: expandedModelConfig = expandedModelConfig()
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    for i in range(config.nn_layers):
        model.add(LSTM(units=config.nn_max_units//(i+1), activation=config.nn_activation, return_sequences=config.nn_return_sequences))
        config.add_dropout(model, (i == config.nn_layers - 1))
    model.add(Dense(units=1))
    model.compile(optimizer=config.optimizer, loss=config.loss)
    return model

class LSTM_model_train:
    def __init__(self, config: modelConfig):
        self.config = config
        self.data : pd.DataFrame = read_yfinance(config.ticker, config.period)
        self.X = []
        self.y = []
        self.model = None

    def _create_dataset(self, scaled_data) -> np:
        X, y = [], []
        for i in range(len(scaled_data) - self.config.time_step - 1):
            X.append(scaled_data[i:(i + self.config.time_step), 0])
            y.append(scaled_data[i + self.config.time_step, 0])
        return np.array(X), np.array(y)

    def _LSTM_parameters(self) -> None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)
        self.X, self.y = self._create_dataset(scaled_data)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)

    def _LSTMModel(self) -> None:
        self.model = create_lstm_model(self.X)

    def LSTMModel_train(self) -> None:
        self._LSTM_parameters()
        self._LSTMModel()
        self.model.fit(self.X, self.y, epochs=self.config.epochs, batch_size=self.config.batch_size)

    def save(self) -> None:
        self.model.save(r'app\\internal\\models\\model_{}.keras'.format(self.config.ticker))