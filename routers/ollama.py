import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, REGISTRY
from prometheus_client.core import GaugeMetricFamily
from app.ollama_client import OllamaClient
from app.cost_tracker import tracker
from app import scheduler as scheduling

router = APIRouter()
client = OllamaClient()

# Cap concurrent requests to Ollama (backpressure), and order the wait queue by
# the configured policy (SCHED_POLICY: fifo|priority|sjf). The cap is unchanged
# from the prior asyncio.Semaphore; the scheduler adds *which-waiter-next*
# control on top of it. See app/scheduler.py.
MAX_CONCURRENT = 2
scheduler = scheduling.from_env(MAX_CONCURRENT)

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

# --- Cost metrics (the thing most gateways never expose to Prometheus) --------
# These mirror the JSON at /metrics/cost into the Prometheus scrape so Grafana
# can chart dollars live, next to latency and throughput. Values are pushed from
# the CostTracker snapshot after each request.
COST_PER_MILLION = Gauge(
    "cost_per_million_tokens",
    "Rolling $/M tokens at the configured GPU hourly rate",
)
COST_PER_REQUEST_P50 = Gauge(
    "cost_per_request_p50_usd",
    "p50 cost per request in USD (rolling window)",
)
COST_PER_REQUEST_P95 = Gauge(
    "cost_per_request_p95_usd",
    "p95 cost per request in USD (rolling window)",
)
COST_SESSION_TOTAL = Gauge(
    "cost_session_total_usd",
    "Cumulative cost since process start in USD",
)
GPU_HOURLY_RATE = Gauge(
    "gpu_hourly_rate_usd",
    "Configured GPU hourly rate (the economic input for all cost math)",
)


def publish_cost_metrics() -> None:
    """Push the CostTracker snapshot into the Prometheus gauges.

    Called after each recorded request so the scrape always reflects the latest
    rolling aggregates without duplicating the cost math.
    """
    snap = tracker.snapshot()
    COST_PER_MILLION.set(snap["cost_per_million_tokens"])
    COST_PER_REQUEST_P50.set(snap["cost_per_request_p50"])
    COST_PER_REQUEST_P95.set(snap["cost_per_request_p95"])
    COST_SESSION_TOTAL.set(snap["total_cost_session"])
    GPU_HOURLY_RATE.set(snap["gpu_hourly_rate"])


# The GPU rate is a constant config input, not a per-request measurement —
# publish it once at startup so the gauge reads correctly even with zero traffic
# (otherwise it sits at the gauge's initial 0 until the first request).
GPU_HOURLY_RATE.set(tracker.gpu_hourly_rate)


class _UtilizationCollector:
    """Reports concurrency-slot utilization *at scrape time* (the correct
    Prometheus pattern for an instantaneous gauge). On a discrete-GPU host you
    would source this from DCGM; here busy-slots / capacity is the proxy, and it
    is what makes the dashboard's utilization-adjusted $/token panel meaningful.
    """

    def collect(self):
        cap = scheduler.max_concurrent or 1
        util = scheduler.active / cap
        g = GaugeMetricFamily(
            "gpu_slots_utilization",
            "Fraction of concurrency slots in use (instantaneous, at scrape)",
        )
        g.add_metric([], util)
        yield g


# Guarded so repeated imports (e.g. under pytest) don't double-register.
try:
    REGISTRY.register(_UtilizationCollector())
except ValueError:
    pass


class ChatRequest(BaseModel):
    model: str = "qwen2.5:7b"
    prompt: str
    max_tokens: int = 256
    # Higher runs sooner under SCHED_POLICY=priority; ignored by fifo. max_tokens
    # doubles as the cost hint SCHED_POLICY=sjf orders by (shortest job first).
    priority: int = 0

@router.post("/chat")
async def chat(request: ChatRequest):
    QUEUE_DEPTH.inc()  # waiting in queue
    async with scheduler.slot(priority=request.priority, cost_hint=request.max_tokens):
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
            publish_cost_metrics()

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

    The scheduler slot and instrumentation live *inside* the generator so the
    slot is held for the full stream lifetime — returning the StreamingResponse
    from the handler does not block, so acquiring outside here would release the
    slot before any tokens were produced.
    """

    async def event_stream():
        QUEUE_DEPTH.inc()  # waiting in queue
        async with scheduler.slot(priority=request.priority, cost_hint=request.max_tokens):
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
                publish_cost_metrics()

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
