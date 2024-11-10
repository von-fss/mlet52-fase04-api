import yfinance as yf
import pandas as pd
import numpy as np
import sklearn.preprocessing
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential, load_model


def train(ticket):
    dfTicket = yf.Ticker(ticket).history(period="1mo")
    dfTicket = dfTicket.drop(["Dividends", "Stock Splits"], axis=1)

    data = pd.DataFrame(dfTicket['Close'])
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 3
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    ############# Update neural layers for next steps (v-0.1) ##########
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, activation='relu', return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=15)
    ####################################################################

    model.save(f'app\\internal\\train_model\\{ticket}.keras')


def predict(ticket):
    load_teste_model = load_model(f'app\\internal\\train_model\\{ticket}.keras')

    dfTicket = yf.Ticker("NVDA").history(period="5d")
    #dfTicket = dfTicket.drop(["Dividends", "Stock Splits"], axis=1)

    data = pd.DataFrame(dfTicket['Close'])
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = 3
    x = []

    for i in range(len(scaled_data) - time_step):
        x.append(scaled_data[i:(i + time_step), 0])

    x = np.array(x)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    prediction = load_teste_model.predict(x)
    prediction = scaler.inverse_transform(prediction)

    return prediction[0][0]
