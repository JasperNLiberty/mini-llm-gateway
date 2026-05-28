from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ollama_client import OllamaClient

router = APIRouter()
client = OllamaClient()

class ChatRequest(BaseModel):
    model: str = "llama3.2:1b"
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = await client.generate(request.model, request.message)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
