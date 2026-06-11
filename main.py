import logging
import os
import time

from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app

from app.cost_tracker import tracker

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gateway")

app = FastAPI()

BACKEND = os.getenv("BACKEND", "ollama")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured per-request access log: method, path, status, latency.

    Health checks and Prometheus scrapes are logged at DEBUG so they don't drown
    out real inference traffic; everything else logs at INFO (errors at WARNING).
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "%s %s -> 500 %.1fms (unhandled)", request.method, request.url.path, duration_ms
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000

    path = request.url.path
    noisy = path in ("/healthz",) or path.startswith("/metrics")
    if noisy:
        level = logging.DEBUG
    elif response.status_code >= 500:
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.log(
        level,
        "%s %s -> %d %.1fms",
        request.method,
        path,
        response.status_code,
        duration_ms,
    )
    return response

@app.get("/healthz")
def healthz():
    return {"status": "ok", "backend": BACKEND}

# Cost metrics. Registered before the Prometheus mount below so this route is
# not shadowed by the /metrics ASGI sub-app.
@app.get("/metrics/cost")
def metrics_cost():
    return tracker.snapshot()

# Load router based on backend
if BACKEND == "ollama":
    from routers import ollama
    app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
elif BACKEND == "mlx":
    from routers import mlx
    app.include_router(mlx.router, prefix="/mlx", tags=["mlx"])

# Mount Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())