from keras.layers import LSTM, Dense, Input, Dropout
from keras.models import Sequential, load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import pandas as pd
from ..routers.utils import modelConfig
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import boto3

def read_yfinance(ticker: str, period: str) -> pd.DataFrame:
    nvda: pd.DataFrame = yf.Ticker(ticker).history(period)
    return pd.DataFrame(nvda['Close'])

class model_prediction:
    def __init__(self, ticker, period):
        self.model = load_model(r'app\internal\models\model_{}.keras'.format(ticker), custom_objects={'rmse_error':rmse_error})
        self.data = read_yfinance(ticker, period)
        self.X = []
        self.time_step = 5

    def predict(self) -> float:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)

        ################### not for mvp ###################
        # Next version check for time_step implementation #
        #for i in range(len(scaled_data)-self.time_step):
        #    self.X.append(scaled_data[i:(i+self.time_step), 0])

        #self.X = np.array(self.X)
        #######################################################
        
        self.X = np.array(scaled_data)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))

        prediction = self.model.predict(self.X)
        prediction = scaler.inverse_transform(prediction[0])
        return prediction[0][0]

    def model_evaluate(self) -> dict:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)
        X = scaled_data[-4:].reshape(1,4,1)
        y = scaled_data[4].reshape(1, 1)
        mse, mae, rmse_error = self.model.evaluate(X, y)
        metric = {
            'MSE':mse,
            'MAE':mae,
            'RMSE':rmse_error
        }
        return metric

@dataclass
class expandedModelConfig(modelConfig):
    def add_dropout(self, model, last_layer) -> None:
        if self.dropout and self.nn_layers > 1 and not last_layer:
            model.add(Dropout(self.dropout_value))

def create_lstm_model(X) -> Sequential:
    config: expandedModelConfig = expandedModelConfig()
    model = Sequential()
    model.add(Input(shape=(np.array(X).shape[1], 1)))
    for i in range(config.nn_layers):
        model.add(LSTM(units=config.nn_max_units//(i+1), activation=config.nn_activation, return_sequences=config.nn_return_sequences))
        config.add_dropout(model, (i == config.nn_layers - 1))
    model.add(Dense(units=1))
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['mae', rmse_error])
    return model

def rmse_error(y_test, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_test - y_pred)))

class LSTM_model_train:
    def __init__(self, config: modelConfig):
        self.config = config
        self.data : pd.DataFrame = read_yfinance(config.ticker, config.period)
        #self.X: np.array = np.array([])
        #self.y: np.array = np.array([])
        self.X_train: np.array = np.array([])
        self.X_test: np.array = np.array([])
        self.y_train: np.array = np.array([])
        self.y_test: np.array = np.array([])
        self.model = None

    def _create_dataset(self, scaled_data) -> None:
        X, y = [], []
        for i in range(len(scaled_data) - self.config.time_step - 1):
            X.append(scaled_data[i:(i + self.config.time_step), 0])
            y.append(scaled_data[i + self.config.time_step, 0])
        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        #return X, y

    def _LSTM_parameters(self) -> None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)  
        self._create_dataset(scaled_data)
        #self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)

    def _LSTM_Model(self) -> None:
        self.model = create_lstm_model(self.X_train)

    def _LSTM_Train(self) -> None:
        self.model.fit(self.X_train, self.y_train, epochs=self.config.epochs, batch_size=self.config.batch_size, validation_data=(self.X_test, self.y_test))

    def LSTMModel_train(self) -> None:
        self._LSTM_parameters()
        self._LSTM_Model()
        self._LSTM_Train()

    def save(self) -> None:
        self.model.save(r'app\\internal\\models\\model_{}.keras'.format(self.config.ticker))
        metric = model_prediction(self.config.ticker, self.config.period).model_evaluate()        
        
        _mse = metric['MSE']
        _mae = metric['MAE']
        _rmse = metric['RMSE']
        
        _tag = f'time_step={self.config.time_step}&epoch={self.config.epochs}&batch_size={self.config.batch_size}&learning_rate={self.config.learning_rate}&nn_activation={self.config.nn_activation}&nn_max_units={self.config.nn_max_units}&MSE={str(_mse)}&MAE={str(_mae)}&RMSE={str(_rmse)}'
        
    # Upload a new file
        s3 = boto3.resource('s3')
        with open(r'app\\internal\\models\\model_{}.keras'.format(self.config.ticker), 'rb') as data:
            s3.Bucket('modeldataqbase').put_object(Key=r'model/{}.keras'.format(self.config.ticker), Body=data, Tagging=_tag)
        