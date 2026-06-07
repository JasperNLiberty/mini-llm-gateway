"""In-memory rolling record of per-request cost, shared across the app.

Keeps the gateway a "measurement instrument": every inference request feeds a
sample in here, and ``/metrics/cost`` reads aggregates back out. State is
process-local and resets on restart, which is exactly what ``total_cost_session``
is meant to express.
"""

import math
import os
from collections import deque
from threading import Lock

import cost_calculator

# GPU_HOURLY_RATE is the swappable economic input. On M1 this stands in for a
# cloud GPU price (e.g. an A10G at ~$0.80/hr) or the amortized local machine.
GPU_HOURLY_RATE = float(os.getenv("GPU_HOURLY_RATE", "0.80"))

# Cap the rolling window so memory stays bounded under sustained load.
MAX_SAMPLES = 1000


class CostTracker:
    def __init__(self, gpu_hourly_rate: float = GPU_HOURLY_RATE,
                 max_samples: int = MAX_SAMPLES):
        self.gpu_hourly_rate = gpu_hourly_rate
        self._costs: deque = deque(maxlen=max_samples)
        self._throughputs: deque = deque(maxlen=max_samples)
        self._total_cost_session = 0.0
        self._requests_observed = 0
        self._lock = Lock()

    def record(self, input_tokens: int, output_tokens: int,
               tokens_per_sec: float) -> float:
        """Record one request and return its USD cost."""
        cost = cost_calculator.cost_per_request(
            self.gpu_hourly_rate, tokens_per_sec, input_tokens, output_tokens
        )
        with self._lock:
            self._costs.append(cost)
            if tokens_per_sec > 0:
                self._throughputs.append(tokens_per_sec)
            self._total_cost_session += cost
            self._requests_observed += 1
        return cost

    def snapshot(self) -> dict:
        """Aggregate view for the /metrics/cost endpoint."""
        with self._lock:
            costs = sorted(self._costs)
            throughputs = list(self._throughputs)
            total = self._total_cost_session
            observed = self._requests_observed

        # Use mean observed throughput for the headline marginal-cost figures;
        # fall back to 0 (calculator returns 0.0) before any traffic.
        avg_tps = sum(throughputs) / len(throughputs) if throughputs else 0.0

        return {
            "cost_per_token": cost_calculator.cost_per_token(
                self.gpu_hourly_rate, avg_tps),
            "cost_per_million_tokens": cost_calculator.cost_per_million_tokens(
                self.gpu_hourly_rate, avg_tps),
            "cost_per_request_p50": _percentile(costs, 50),
            "cost_per_request_p95": _percentile(costs, 95),
            "total_cost_session": total,
            "gpu_hourly_rate": self.gpu_hourly_rate,
            "requests_observed": observed,
        }


def _percentile(sorted_values: list, pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    rank = math.ceil((pct / 100.0) * len(sorted_values))
    k = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[k]


# Process-wide singleton shared by the router and the metrics endpoint.
tracker = CostTracker()
