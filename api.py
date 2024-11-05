from fastapi import FastAPI
from fastapi.responses import JSONResponse
from api_help.get_data import get_data

app = FastAPI()

@app.get("/yfinance/")
def yfinance():
    return JSONResponse(content=get_data())