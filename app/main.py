import time

from fastapi import FastAPI, Request

from .routers import model, yfinance
from pyinstrument import Profiler, renderers
from memory_profiler import memory_usage

app = FastAPI()

app.include_router(model.router)
app.include_router(yfinance.router)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    start_mem = memory_usage()[0]
    profiler = Profiler(interval=0.001, async_mode="enabled")
    profiler.start()
    response = await call_next(request)
    end_mem = memory_usage()[0]
    process_time = time.perf_counter() - start_time
    profiler.stop()
    stats = profiler.last_session

    response.headers["X-CPU-Time"] = str(stats.cpu_time)
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Memory-Usage"] = str(end_mem - start_mem)

    return response
    #return await call_next(request)



@app.get('/')
async def root():
    return {'message': 'Hello Bigget Applications'}



