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

# @router.get('/')
# async def read_yfinance(ticker: str):
#     """
#     Recebe o código de um ticker e retorna o histórico de 1mês

#     :param ticker: Código do Ticker

#     :return: Os valores do último mês
#     """
#     nvda = yf.Ticker(ticker).history(period="1mo").reset_index()
#     nvda['Date'] = pd.to_datetime(nvda['Date']).dt.strftime('%Y-%m-%d')
#     return JSONResponse(nvda.to_dict(orient="records"))


@router.get('/tickers', response_model=List[ModelTickers])
async def list_tickers() -> List[dict[str, str]]:
    """
    Return the list of all available tickets

    :return: dictionary containing all tickets with keys "names"
    """
    ticker = get_tickers_list()['symbol'].unique()    
    return [{"name": symbol} for symbol in ticker]

    # return [
    #     ModelTickers(name="TSLA"),
    #     ModelTickers(name="NVDA"),
    #     ModelTickers(name="DJT"),
    #     ModelTickers(name="LCID"),
    #     ModelTickers(name="PLTR"),
    #     ModelTickers(name="SOFI"),
    #     ModelTickers(name="INTC"),
    #     ModelTickers(name="IONQ"),
    #     ModelTickers(name="SMCI"),
    #     ModelTickers(name="VALE"),
    #     ModelTickers(name="RIVN"),
    #     ModelTickers(name="NIO"),
    #     ModelTickers(name="PINS"),
    #     ModelTickers(name="PFE"),
    #     ModelTickers(name="SOUN"),
    #     ModelTickers(name="WBD"),
    #     ModelTickers(name="GSAT"),
    #     ModelTickers(name="MARA"),
    #     ModelTickers(name="F"),
    #     ModelTickers(name="VLY"),
    #     ModelTickers(name="UPST"),
    #     ModelTickers(name="SNAP"),
    #     ModelTickers(name="BAC"),
    #     ModelTickers(name="AAPL"),
    #     ModelTickers(name="CLSK"),
    #     ModelTickers(name="AMZN"),
    #     ModelTickers(name="ABEV"),
    #     ModelTickers(name="DKNG"),
    #     ModelTickers(name="T"),
    #     ModelTickers(name="RIOT"),
    #     ModelTickers(name="RUN"),
    #     ModelTickers(name="LYFT"),
    #     ModelTickers(name="TOST"),
    #     ModelTickers(name="PTON"),
    #     ModelTickers(name="GOLD"),
    #     ModelTickers(name="AMD"),
    #     ModelTickers(name="IOVA"),
    #     ModelTickers(name="NU"),
    #     ModelTickers(name="AFRM"),
    #     ModelTickers(name="WULF"),
    #     ModelTickers(name="ET"),
    #     ModelTickers(name="BABA"),
    #     ModelTickers(name="SQ"),
    #     ModelTickers(name="SMR"),
    #     ModelTickers(name="U"),
    #     ModelTickers(name="BBD"),
    #     ModelTickers(name="LUMN"),
    #     ModelTickers(name="BE"),
    #     ModelTickers(name="CMCSA"),
    #     ModelTickers(name="CNH"),
    #     ModelTickers(name="HOOD"),
    #     ModelTickers(name="JBLU"),
    #     ModelTickers(name="VZ"),
    #     ModelTickers(name="WBA"),
    #     ModelTickers(name="GRAB"),
    #     ModelTickers(name="KGC"),
    #     ModelTickers(name="RIG"),
    #     ModelTickers(name="MPW"),
    #     ModelTickers(name="AGNC"),
    #     ModelTickers(name="AAL"),
    #     ModelTickers(name="OKLO"),
    #     ModelTickers(name="UBER"),
    #     ModelTickers(name="AES"),
    #     ModelTickers(name="ITUB"),
    #     ModelTickers(name="PBR"),
    #     ModelTickers(name="ABNB"),
    #     ModelTickers(name="CCL"),
    #     ModelTickers(name="MSTR"),
    #     ModelTickers(name="XPEV"),
    #     ModelTickers(name="UAA"),
    #     ModelTickers(name="NOK"),
    #     ModelTickers(name="HIMS"),
    #     ModelTickers(name="RKLB"),
    #     ModelTickers(name="HL"),
    #     ModelTickers(name="IQ"),
    #     ModelTickers(name="CSCO"),
    #     ModelTickers(name="COIN"),
    #     ModelTickers(name="KMI"),
    #     ModelTickers(name="MSFT"),
    #     ModelTickers(name="KVUE"),
    #     ModelTickers(name="FCX"),
    #     ModelTickers(name="IREN"),
    #     ModelTickers(name="GME"),
    #     ModelTickers(name="PARA"),
    #     ModelTickers(name="HBAN"),
    #     ModelTickers(name="ALTM"),
    #     ModelTickers(name="BTG"),
    #     ModelTickers(name="CX"),
    #     ModelTickers(name="KEY"),
    #     ModelTickers(name="CLF"),
    #     ModelTickers(name="PDD"),
    #     ModelTickers(name="GOOGL"),
    #     ModelTickers(name="PCG"),
    #     ModelTickers(name="GGB"),
    #     ModelTickers(name="BEKE"),
    #     ModelTickers(name="APP"),
    #     ModelTickers(name="GOOG"),
    #     ModelTickers(name="ERIC"),
    #     ModelTickers(name="C"),
    #     ModelTickers(name="KO")
    # ]
