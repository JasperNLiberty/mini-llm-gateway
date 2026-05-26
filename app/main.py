from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.ollama_client import OllamaClient

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.client = OllamaClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.close()

class ChatRequest(BaseModel):
    model: str = "llama3.2"
    prompt: str

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatRequest):
    result = await app.state.client.generate(request.model, request.prompt)
    return {"response": result}
