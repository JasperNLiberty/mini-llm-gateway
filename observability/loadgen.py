#!/usr/bin/env python3
"""Fire mixed-size requests at the gateway to populate the Grafana dashboard.

Stdlib only (no extra deps). Varies prompt size so the cost, latency, throughput,
and queue-depth panels all show movement — including some queueing past the
gateway's concurrency cap.

    python observability/loadgen.py --duration 60 --concurrency 4
"""

import argparse
import json
import random
import threading
import time
import urllib.request

# (prompt, max_tokens) — a spread of short/long generations
PROMPTS = [
    ("say hello in one word", 16),
    ("summarize the plot of The Matrix in one sentence", 64),
    ("write a short python function that computes factorial", 128),
    ("explain how transformer attention works, step by step", 256),
]


def worker(stop_at, url, model, tally, lock):
    while time.time() < stop_at:
        prompt, max_tokens = random.choice(PROMPTS)
        body = json.dumps(
            {"model": model, "prompt": prompt, "max_tokens": max_tokens}
        ).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        key = "ok"
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
        except Exception:
            key = "err"
        with lock:
            tally[key] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    url = args.gateway.rstrip("/") + "/ollama/chat"
    stop_at = time.time() + args.duration
    tally = {"ok": 0, "err": 0}
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(stop_at, url, args.model, tally, lock))
        for _ in range(args.concurrency)
    ]
    print(f"load: {args.concurrency} workers x {args.duration}s -> {url} ({args.model})")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {tally['ok']} ok, {tally['err']} errors")


if __name__ == "__main__":
    main()
