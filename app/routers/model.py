from fastapi import APIRouter
from app.internal.train_model.train import train_model
from app.internal.predict import predict_model

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {'description': 'Not Found'}}
)

@router.get('/train')
def train_model_route(ticker: str) -> dict:
    """
    Description:
        Recebe o código de um ticker, treina o modelo e salva na pasta train_model

    Args:
        ticker: str -> Código do ticker a ser analisado
    """
    train_model(ticker)
    return {'result': f'modelo {ticker} treinado com sucesso'}


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
