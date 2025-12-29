from fastapi import FastAPI, Request

from src.api.metrics import REQUEST_COUNT, metrics

app = FastAPI()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    REQUEST_COUNT.labels(
        request.method,
        request.url.path,
        response.status_code,
    ).inc()
    return response


@app.get("/metrics")
def prometheus_metrics():
    return metrics()
