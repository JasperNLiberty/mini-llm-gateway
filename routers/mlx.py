from fastapi import APIRouter
from pydantic import BaseModel
from mlx_lm import load, generate
import asyncio
from collections import deque
from typing import List
import numpy as np

router = APIRouter()

model, tokenizer = load('Qwen/Qwen2.5-7B')

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

# Batching config
BATCH_SIZE = 32
BATCH_TIMEOUT = 0.1  # seconds

request_queue: deque = deque()
batch_lock = asyncio.Lock()

async def process_batch():
    """Process queued requests in batches"""
    while True:
        await asyncio.sleep(BATCH_TIMEOUT)
        
        async with batch_lock:
            if not request_queue:
                continue
            
            # Collect batch
            batch = []
            for _ in range(min(BATCH_SIZE, len(request_queue))):
                batch.append(request_queue.popleft())
            
            if not batch:
                continue
            
            # Extract prompts and futures
            prompts = [req.prompt for req, _ in batch]
            futures = [future for _, future in batch]
            
            # Process batch with MLX
            try:
                responses = []
                for prompt in prompts:
                    response = generate(
                        model, 
                        tokenizer, 
                        prompt=prompt, 
                        max_tokens=batch[0][0].max_tokens
                    )
                    responses.append(response)
                
                # Return results to each client
                for future, response in zip(futures, responses):
                    if not future.done():
                        future.set_result({"response": response})
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

@router.post("/chat")
async def chat(req: ChatRequest):
    future = asyncio.Future()
    request_queue.append((req, future))
    return await future

# Start batch processor on startup
@router.on_event("startup")
async def startup():
    asyncio.create_task(process_batch())