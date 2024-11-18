from fastapi import APIRouter
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
from typing import List
from .utils import get_tickers_list, Ticker as ModelTickers

router = APIRouter(
    prefix="/yfinance",
    tags=["yfinance"],
    responses={404: {'description': 'Not Found'}}
)

@router.get('/getHistory')
async def read_yfinance(ticker: str):
#     """
#     Recebe o código de um ticker e retorna o histórico de 1mês

#     :param ticker: Código do Ticker

#     :return: Os valores do último mês
#     """
    nvda = yf.Ticker(ticker).history(period="1mo").reset_index()
    nvda['Date'] = pd.to_datetime(nvda['Date']).dt.strftime('%Y-%m-%d')
    return JSONResponse(nvda.to_dict(orient="records"))


@router.get('/tickers', response_model=List[ModelTickers])
async def list_tickers() -> List[dict[str, str]]:
    """
    Return the list of all available tickets

    :return: dictionary containing all tickets with keys "names"
    """
    ticker = get_tickers_list()['symbol'].unique()    
    return [{"name": symbol} for symbol in ticker]