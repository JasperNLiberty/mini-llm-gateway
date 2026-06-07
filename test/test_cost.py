"""Tests for the cost instrumentation layer.

Runs with pytest if available, or standalone: ``python test/test_cost.py``.
No third-party test deps required.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cost_calculator
from app.cost_tracker import CostTracker, _percentile


def test_cost_per_token():
    # 0.80 / (100 * 3600)
    assert abs(cost_calculator.cost_per_token(0.80, 100.0) - 2.2222e-6) < 1e-10


def test_cost_per_token_zero_throughput_is_safe():
    assert cost_calculator.cost_per_token(0.80, 0) == 0.0
    assert cost_calculator.cost_per_token(0.80, -5) == 0.0


def test_cost_per_million_tokens_scales_by_million():
    cpt = cost_calculator.cost_per_token(0.80, 100.0)
    assert abs(cost_calculator.cost_per_million_tokens(0.80, 100.0) - cpt * 1_000_000) < 1e-12


def test_cost_per_request_sums_input_and_output():
    cpt = cost_calculator.cost_per_token(0.80, 100.0)
    expected = cpt * (45 + 128)
    assert abs(cost_calculator.cost_per_request(0.80, 100.0, 45, 128) - expected) < 1e-15


def test_effective_cost_per_token():
    assert cost_calculator.effective_cost_per_token(2e-6, 0.5) == 4e-6
    assert cost_calculator.effective_cost_per_token(2e-6, 1.0) == 2e-6
    assert cost_calculator.effective_cost_per_token(2e-6, 0) == 0.0


def test_percentile_nearest_rank():
    data = sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert _percentile(data, 50) == 5
    assert _percentile(data, 95) == 10
    assert _percentile([], 50) == 0.0


def test_tracker_accumulates_session_and_counts():
    t = CostTracker(gpu_hourly_rate=0.80)
    c1 = t.record(input_tokens=10, output_tokens=20, tokens_per_sec=50.0)
    c2 = t.record(input_tokens=5, output_tokens=15, tokens_per_sec=50.0)
    snap = t.snapshot()
    assert snap["requests_observed"] == 2
    assert abs(snap["total_cost_session"] - (c1 + c2)) < 1e-15
    assert snap["gpu_hourly_rate"] == 0.80
    assert snap["cost_per_million_tokens"] > 0


def test_tracker_empty_snapshot_is_zeroed_not_error():
    snap = CostTracker().snapshot()
    assert snap["requests_observed"] == 0
    assert snap["total_cost_session"] == 0.0
    assert snap["cost_per_token"] == 0.0
    assert snap["cost_per_request_p50"] == 0.0


def test_metrics_cost_route_not_shadowed_by_prometheus_mount():
    # /metrics is a mounted ASGI sub-app; ensure /metrics/cost still resolves
    # to our JSON handler rather than the Prometheus exporter.
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    resp = client.get("/metrics/cost")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("cost_per_token", "cost_per_million_tokens",
                "cost_per_request_p50", "cost_per_request_p95",
                "total_cost_session", "gpu_hourly_rate", "requests_observed"):
        assert key in body


def _run_standalone():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
