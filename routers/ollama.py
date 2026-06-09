import json
import time
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming variant of /chat. Emits NDJSON: one line per token delta,
    then a final line (``done: true``) carrying usage, timings, and cost.

    The semaphore and instrumentation live *inside* the generator so the slot
    is held for the full stream lifetime — returning the StreamingResponse from
    the handler does not block, so acquiring outside here would release the slot
    before any tokens were produced.
    """

    async def event_stream():
        QUEUE_DEPTH.inc()  # waiting in queue
        async with queue_semaphore:
            QUEUE_DEPTH.dec()  # left queue, now processing
            IN_FLIGHT.inc()
            start_time = time.time()
            first_token_time = None
            response_text = ""
            input_tokens = 0
            output_tokens = 0
            tokens_per_sec = 0.0
            try:
                async for chunk in client.generate_stream(
                    request.model, request.prompt, request.max_tokens
                ):
                    if chunk.get("done"):
                        # Final chunk: authoritative usage + generation timing.
                        output_tokens = chunk.get("eval_count", 0)
                        input_tokens = chunk.get("prompt_eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
                        tokens_per_sec = (
                            output_tokens / (eval_duration_ns / 1e9)
                            if eval_duration_ns > 0
                            else 0.0
                        )
                        continue

                    delta = chunk.get("response", "")
                    if delta and first_token_time is None:
                        first_token_time = time.time()
                    response_text += delta
                    yield json.dumps(
                        {"delta": delta, "t": time.time() - start_time}
                    ) + "\n"

                elapsed = time.time() - start_time
                ttft = (first_token_time - start_time) if first_token_time else None

                REQUESTS.labels(model=request.model, status="success").inc()
                LATENCY.labels(model=request.model).observe(elapsed)
                TOKENS.labels(model=request.model).inc(output_tokens)
                cost_usd = tracker.record(input_tokens, output_tokens, tokens_per_sec)

                yield json.dumps(
                    {
                        "done": True,
                        "response": response_text,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_sec": tokens_per_sec,
                        "ttft": ttft,
                        "elapsed": elapsed,
                        "cost_usd": cost_usd,
                    }
                ) + "\n"
            except Exception as e:
                REQUESTS.labels(model=request.model, status="error").inc()
                yield json.dumps({"error": str(e)}) + "\n"
            finally:
                IN_FLIGHT.dec()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
