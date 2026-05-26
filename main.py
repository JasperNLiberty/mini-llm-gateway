from fastapi import FastAPI
from routers import ollama

app = FastAPI()

app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
