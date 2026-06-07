import time
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge
from app.ollama_client import OllamaClient
from app.cost_tracker import tracker

router = APIRouter()
client = OllamaClient()

# Limit concurrent requests to Ollama
MAX_CONCURRENT = 2
queue_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Prometheus metrics
REQUESTS = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["model", "status"],
)

LATENCY = Histogram(
    "chat_latency_seconds",
    "Request latency in seconds",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
)

TOKENS = Counter(
    "tokens_generated_total",
    "Total tokens generated",
    ["model"],
)

IN_FLIGHT = Gauge(
    "chat_in_flight",
    "Number of requests currently being processed",
)

QUEUE_DEPTH = Gauge(
    "chat_queue_depth",
    "Number of requests waiting in queue",
)


class ChatRequest(BaseModel):
    model: str = "qwen2.5:7b"
    prompt: str
    max_tokens: int = 256

@router.post("/chat")
async def chat(request: ChatRequest):
    QUEUE_DEPTH.inc()  # waiting in queue
    async with queue_semaphore:
        QUEUE_DEPTH.dec()  # left queue, now processing
        IN_FLIGHT.inc()
        start_time = time.time()
        try:
            result = await client.generate(request.model, request.prompt, request.max_tokens)
            elapsed = time.time() - start_time

            REQUESTS.labels(model=request.model, status="success").inc()
            LATENCY.labels(model=request.model).observe(elapsed)
            TOKENS.labels(model=request.model).inc(result["tokens"])

            cost_usd = tracker.record(
                result["input_tokens"],
                result["output_tokens"],
                result["tokens_per_sec"],
            )

            return {
                "response": result["response"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "tokens_per_sec": result["tokens_per_sec"],
                "cost_usd": cost_usd,
            }
        except Exception as e:
            REQUESTS.labels(model=request.model, status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            IN_FLIGHT.dec()
