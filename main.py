from fastapi import FastAPI
from prometheus_client import make_asgi_app
from routers import ollama

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])

# Mount Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())
