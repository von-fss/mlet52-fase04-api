from .utils import model_prediction

def predict_model(ticker) -> float:
    yfinance_prediction = model_prediction(ticker, '5d')
    return yfinance_prediction.predict()

def evaluate_model(ticker) -> dict:
    yfinance_prediction = model_prediction(ticker, '5d')
    return yfinance_prediction.model_evaluate()