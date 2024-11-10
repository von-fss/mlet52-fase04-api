import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

load_yfinance_model = load_model(r'app\internal\train_model\models\yfinance_nvda.keras')

def read_yfinance():
    nvda = yf.Ticker("NVDA").history(period="5d")#.reset_index()
    # nvda['Date'] = pd.to_datetime(nvda['Date']).dt.date
    nvda = nvda.drop(["Dividends", "Stock Splits"], axis=1)
    return nvda

data = pd.DataFrame(read_yfinance()['Close'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

time_step = 3
X = []

for i in range(len(scaled_data)-time_step):
    X.append(scaled_data[i:(i+time_step), 0])

X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

prediction = load_yfinance_model.predict(X)
prediction = scaler.inverse_transform(prediction)
print(prediction)