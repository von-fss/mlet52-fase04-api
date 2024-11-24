from fastapi import APIRouter
from ..internal.train import train_model
from ..internal.predict import predict_model, evaluate_model
from .utils import modelConfig
from ..internal.utils import model_prediction

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {'description': 'Not Found'}}
)


### Deixar os hyperparameters de maior relevância travados (optimizer, activation e return_sequences)
### time_step tem que ser o mesmo time_step da predição ###

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
        loss: Optional[str] = 'mean_squared_error',  # Optional parameter with default value 'mean_squared_error'
        dropout: Optional[bool] = True,  # Optional parameter with default value True
        dropout_value: Optional[float] = 0.2  # Optional parameter with default value 0.2        
    """
    train_model(config)
    return {'result': config.get_parameters()}

@router.get('/predict')
def predict_model_route(ticker: str) -> dict:
    """
    Description:
        Predict next value from a pre trained model
    Args:
        ticker: str -> Código do ticker a ser analisado
    """
    prediction = predict_model(ticker)
    return {"ticker": str(ticker), "predicted": float(prediction)}

@router.get('/list')
def list_model_route() -> tuple:
    return {(1, 2)}
    
@router.get('/evaluate')
def evaluate_model_route(ticker: str) -> dict:
    return evaluate_model(ticker)


@router.get('/get_evaluate_from_training')
def load_evaluate_from_training(ticker: str) -> dict:
    return model_prediction.get_evaluate_from_training()