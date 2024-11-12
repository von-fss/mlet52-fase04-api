from fastapi import APIRouter
from app.internal.train_model.train import train_model
from app.internal.predict import predict_model
from typing import Optional
from ..internal.train_model.utils import modelConfig

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {'description': 'Not Found'}}
)

@router.post('/train')
async def train_model_route(config: modelConfig) -> dict:
    """
    Description:
        Receives the ticker code, trains the model, and saves it in the train_model folder.
        All model parameter is configurable and have default values applied

    Args:
        ticker: Optional[str] = NVDA -> Código do ticker a ser analisado
        time_step: Optional[int] = 5,  # Optional parameter with default value 5
        epochs: Optional[int] = 20,  # Optional parameter with default value 20
        optimizer: Optional[str] = 'adam',  # Optional parameter with default value 'adam'
        batch_size: Optional[int] = 15,  # Optional parameter with default value 15
        learning_rate: Optional[float] = 0.05,  # Optional parameter with default value 0.05
        nn_activation: Optional[str] = 'relu',  # Optional parameter with default value 'relu'
        nn_max_units: Optional[int] = 100,  # Optional parameter with default value 100
        nn_layers: Optional[int] = 2,  # Optional parameter with default value 2
        nn_return_sequences: Optional[bool] = False,  # Optional parameter with default value False
        loss: Optional[str] = 'mean_squared_error',  # Optional parameter with default value 'mean_squared_error'
        dropout: Optional[bool] = True,  # Optional parameter with default value True
        dropout_value: Optional[float] = 0.2  # Optional parameter with default value 0.2        
    """
    train_model(config)
    return {'result': config.get_parameters()}


# @router.get('/predict')
# def predict_model(ticker: str) -> dict:
#     """
#     Description:
#         Procura por um modelo já gerado e fazer a predição

#     Args:
#         ticker: str -> Código do ticker a ser analisado
#     """
#     prediction = predict_model(ticker)
#     return {"ticker": str(ticker), "predicted": float(prediction)}
