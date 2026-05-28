from fastapi import FastAPI
from routers import ollama

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
