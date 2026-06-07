# mini-llm-gateway

A minimal FastAPI inference gateway that sits in front of a local model server
(Ollama on Apple M1, Metal/MPS) and exposes a clean HTTP API for inference —
while treating cost as a first-class, measured quantity.

Most gateways report latency and request counts. This one also reports **what
inference actually costs in dollars**, live, per request and in aggregate.

## Running

```bash
pip install -r requirements.txt
GPU_HOURLY_RATE=0.80 uvicorn main:app --reload
```

`BACKEND` selects the backend (`ollama`, default). `GPU_HOURLY_RATE` sets the
economic input (USD/hour) used for all cost math; default `0.80`.

## Inference

```bash
curl -s localhost:8000/ollama/chat \
  -d '{"model":"qwen2.5:7b","prompt":"count to 5","max_tokens":50}'
```

Every inference response carries cost metadata alongside the output:

```json
{
  "response": "...",
  "input_tokens": 45,
  "output_tokens": 128,
  "tokens_per_sec": 73.2,
  "cost_usd": 0.0000019
}
```

## `GET /metrics/cost`

Returns live, session-wide cost aggregates:

```json
{
  "cost_per_token": 0.0000000030,
  "cost_per_million_tokens": 3.04,
  "cost_per_request_p50": 0.0000012,
  "cost_per_request_p95": 0.0000041,
  "total_cost_session": 0.0009123,
  "gpu_hourly_rate": 0.80,
  "requests_observed": 412
}
```

`cost_per_*` figures are derived from observed throughput at the configured
`GPU_HOURLY_RATE`; percentiles come from a rolling in-memory window of recent
requests; `total_cost_session` accumulates from process start.

**Why `$/M tokens` is a first-class metric.** Throughput (tokens/sec) and
latency tell you how *fast* a server is, but not whether it's *economical* — a
fast GPU that costs 10× as much can be the worse choice. Cost per million tokens
collapses hardware price and real throughput into the single unit the industry
quotes prices in, making serving options directly comparable. Exposing it live
turns the gateway from a proxy into an observability instrument: you can see the
dollar impact of a model swap, a batch-size change, or idle GPU time the moment
it happens.

> Prometheus operational metrics (request counts, latency, queue depth) remain
> available at `/metrics`.
