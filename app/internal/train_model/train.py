import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential

def read_yfinance():
    nvda = yf.Ticker("NVDA").history(period="1mo")
    nvda = nvda.drop(["Dividends", "Stock Splits"], axis=1)
    return nvda

data = pd.DataFrame(read_yfinance()['Close'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
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

model.save(r'app\internal\train_model\models\yfinance_nvda.keras')