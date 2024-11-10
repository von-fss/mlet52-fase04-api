from fastapi import APIRouter
from app.internal.model import train, predict

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {'description': 'Not Found'}}
)


@router.get('/train')
def train_model(ticker: str):
    """
    Description:
        Recebe o código de um ticker, treina o modelo e salva na pasta train_model

    Args:
        ticker: str -> Código do ticker a ser analisado
    """
    train(ticker)
    return {"result": 'modelo treinado com sucesso'}


@router.post('/predict')
def value_model(ticker: str):
    """
    Description:
        Procura por um modelo já gerado e fazer a predição

    Args:
        ticker: str -> Código do ticker a ser analisado
    """
    prediction = predict(ticker)
    return {"ticker": str(ticker), "predicted": float(prediction)}
