import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf

## Mocked
yfinance_config: dict = {
    'ticker':'NVDA',
    'period':'1mo'    
}
#################

def read_yfinance(papper: str, period: str) -> pd.DataFrame:
    nvda: pd.DataFrame = yf.Ticker(papper).history(period)
    return pd.DataFrame(nvda['Close'])

class model_prediction:
    def __init__(self, papper, period):
        self.model = load_model(r'app\internal\train_model\models\model_{}.keras'.format(papper))
        self.data = read_yfinance(papper, period)
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