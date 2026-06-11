import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app

from app.cost_tracker import tracker

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",   # the message is already a JSON line
)
logger = logging.getLogger("gateway")

app = FastAPI()

BACKEND = os.getenv("BACKEND", "ollama")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured (JSON) per-request access log with a correlation id.

    Emits one JSON object per request -- queryable by log tooling (Loki, jq,
    CloudWatch) rather than just greppable. Each request gets a request_id, also
    returned as the X-Request-ID response header so a client log line can be
    joined to this server log line. Health checks and Prometheus scrapes log at
    DEBUG so they don't drown out real inference traffic; 5xx logs at WARNING.
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()

    def emit(level: int, status: int) -> None:
        logger.log(level, json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "backend": BACKEND,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }))

    try:
        response = await call_next(request)
    except Exception:
        emit(logging.ERROR, 500)
        raise

    path = request.url.path
    if path in ("/healthz",) or path.startswith("/metrics"):
        level = logging.DEBUG
    elif response.status_code >= 500:
        level = logging.WARNING
    else:
        level = logging.INFO
    emit(level, response.status_code)
    response.headers["X-Request-ID"] = request_id
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