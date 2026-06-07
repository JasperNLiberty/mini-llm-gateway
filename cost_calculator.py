"""Pure, dependency-free cost math for LLM inference.

This module is the economic core of the gateway. It is intentionally free of
any framework/runtime dependency so it can be reused verbatim by the sibling
``llm-serving-benchmarks`` repo. Everything here is a pure function of its
arguments.

The central idea: an inference server costs money per wall-clock hour
(``gpu_hourly_rate``), and produces tokens at some throughput
(``tokens_per_sec``). Divide the two and you get the marginal cost of a token.
The headline portfolio metric is cost per *million* tokens, because that is the
unit the rest of the industry quotes.

Run the doctests with::

    python -m doctest cost_calculator.py -v
"""

SECONDS_PER_HOUR = 3600
TOKENS_PER_MILLION = 1_000_000


def cost_per_token(gpu_hourly_rate: float, tokens_per_sec: float) -> float:
    """USD cost of producing one token at a given throughput.

    ``cost_per_token = gpu_hourly_rate / (tokens_per_sec * 3600)``

    >>> round(cost_per_token(0.80, 100.0), 12)
    2.222222e-06
    >>> cost_per_token(0.80, 0)
    0.0
    """
    if tokens_per_sec <= 0:
        return 0.0
    return gpu_hourly_rate / (tokens_per_sec * SECONDS_PER_HOUR)


def cost_per_million_tokens(gpu_hourly_rate: float, tokens_per_sec: float) -> float:
    """USD cost per 1,000,000 tokens. The headline portfolio metric.

    >>> round(cost_per_million_tokens(0.80, 100.0), 6)
    2.222222
    >>> cost_per_million_tokens(0.80, 0)
    0.0
    """
    return cost_per_token(gpu_hourly_rate, tokens_per_sec) * TOKENS_PER_MILLION


def cost_per_request(gpu_hourly_rate: float, tokens_per_sec: float,
                     input_tokens: int, output_tokens: int) -> float:
    """USD cost of a single request given its token counts.

    Cost is driven by the total tokens the server had to move through it
    (prompt + completion) at the observed throughput.

    >>> round(cost_per_request(0.80, 100.0, 45, 128), 10)
    0.0003844444
    >>> cost_per_request(0.80, 0, 45, 128)
    0.0
    """
    total_tokens = input_tokens + output_tokens
    return cost_per_token(gpu_hourly_rate, tokens_per_sec) * total_tokens


def effective_cost_per_token(nominal_cost_per_token: float, utilization: float) -> float:
    """Adjust nominal cost for real-world GPU utilization.

    ``effective = nominal / utilization`` (utilization in ``(0, 1]``).

    Idle GPU time still costs money, so the real cost of a token is higher than
    the theoretical cost when the GPU is not fully saturated.

    >>> effective_cost_per_token(2e-06, 1.0)
    2e-06
    >>> effective_cost_per_token(2e-06, 0.5)
    4e-06
    >>> effective_cost_per_token(2e-06, 0)
    0.0
    """
    if utilization <= 0:
        return 0.0
    return nominal_cost_per_token / utilization


if __name__ == "__main__":
    import doctest

    failures, _ = doctest.testmod(verbose=False)
    if not failures:
        print("All cost_calculator doctests passed.")
