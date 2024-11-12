from keras.layers import LSTM, Dense, Input, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from ..utils import read_yfinance
from dataclasses import dataclass

@dataclass
class modelConfig:
    ticker: str = 'NVDA'
    time_step: int = 5
    epochs: int = 20
    optimizer: str = 'adam'
    batch_size: int = 15
    learning_rate: float = 0.05
    nn_activation: str = 'relu'
    nn_max_units: int = 100
    nn_layers: int = 2
    nn_return_sequences: bool = False
    loss: str = 'mean_squared_error'
    dropout: bool = True
    dropout_value: float = 0.2
    period: str = '3mo'

    def get_parameters(self) -> dict:
        return {
            "time_step": self.time_step,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "nn_activation": self.nn_activation,
            "nn_max_units": self.nn_max_units,
            "nn_layers": self.nn_layers,
            "nn_return_sequences": self.nn_return_sequences,
            "loss": self.loss,
            "dropout": self.dropout,
            "dropout_value": self.dropout_value
        }

    def add_dropout(self, model, last_int):
        if self.dropout and self.nn_layers > 1 and not last_int:
            model.add(Dropout(self.dropout_value))

def create_lstm_model(config: modelConfig, X) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    for i in range(config.nn_layers):
        model.add(LSTM(units=config.nn_max_units//(i+1), activation=config.nn_activation, return_sequences=config.nn_return_sequences))
        config.add_dropout(model, i == config.nn_layers - 1)
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
        time_step = self.config.time_step
        self.X, self.y = self._create_dataset(scaled_data, time_step)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)

    def _LSTMModel(self) -> None:
        self.mode = create_lstm_model(self.X)

    def LSTMModel_train(self) -> None:
        self._LSTM_parameters()
        self._LSTMModel()
        self.model.fit(self.X, self.y, epochs=self.config.epochs, batch_size=self.config.batch_size)

    def save(self) -> None:
        self.model.save(r'app\\internal\\train_model\\models\\model_{}.keras'.format(self.config.ticker))