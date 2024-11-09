from fastapi import FastAPI
from .routers import model, yfinance

app = FastAPI()

app.include_router(model.router)
app.include_router(yfinance.router)

@app.get('/')
async def root():
    return {'message': 'Hello Bigget Applications'}