import time

from fastapi import FastAPI, Request
from .routers import model, yfinance

app = FastAPI()

app.include_router(model.router)
app.include_router(yfinance.router)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    print(response)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response



@app.get('/')
async def root():
    return {'message': 'Hello Bigget Applications'}



