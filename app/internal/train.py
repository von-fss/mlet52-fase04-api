import pandas as pd
from .utils import LSTM_model_train
from ..routers.utils import modelConfig

def train_model(config: modelConfig) -> None:
    lstm_model = LSTM_model_train(config)
    lstm_model.LSTMModel_train()
    # lstm_model.save()

##### thinking about it
# from .utils import modelConfig
# def get_parameters(config: modelConfig) -> dict:
#     return config.get_parameters()
    