import os
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.cost_tracker import tracker

app = FastAPI()

BACKEND = os.getenv("BACKEND", "ollama")

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