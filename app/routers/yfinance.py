from fastapi import APIRouter
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd

router = APIRouter(
    prefix="/yfinance",
    tags=["yfinance"],
    responses={404: { 'description': 'Not Found'}}
)

@router.get('/')
async def read_yfinance():
    nvda = yf.Ticker("NVDA").history(period="1mo").reset_index()
    nvda['Date'] = pd.to_datetime(nvda['Date']).dt.strftime('%Y-%m-%d')    
    return JSONResponse(nvda.to_dict(orient="records"))