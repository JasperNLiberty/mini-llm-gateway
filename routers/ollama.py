from fastapi import APIRouter, HTTPException
import requests

router = APIRouter()

@router.post("/chat")
async def chat(message: str):
    try:
        response = requests.post("http://ollama-server/chat", json={"message": message})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream")
async def stream():
    try:
        response = requests.get("http://ollama-server/stream", stream=True)
        response.raise_for_status()
        return StreamingResponse(response.iter_content(chunk_size=None))
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def metrics():
    try:
        response = requests.get("http://ollama-server/metrics")
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/healthz")
async def healthz():
    try:
        response = requests.get("http://ollama-server/healthz")
        response.raise_for_status()
        return {"status": "ok"}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
