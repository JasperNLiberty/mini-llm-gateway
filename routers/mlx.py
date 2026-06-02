from fastapi import APIRouter
from pydantic import BaseModel
from mlx_lm import load, generate

router = APIRouter()

# Load model once at startup
model, tokenizer = load('mlx-community/Llama-3.2-3B-Instruct-4bit')

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@router.post("/chat")
async def chat(req: ChatRequest):
    response = generate(model, tokenizer, prompt=req.prompt, max_tokens=req.max_tokens)
    return {"response": response}