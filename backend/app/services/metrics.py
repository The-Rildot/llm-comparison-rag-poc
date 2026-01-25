import statistics

def calculate_metrics(
    response: str,
    latency_ms: float,
    model_key: str
):
    tokens = max(1, len(response) // 4)

    tps = round(tokens / (latency_ms / 1000), 2)

    pricing = {
        "gpt": 0.00000015,
        "llama": 0.0,
        "mistral": 0.0
    }

    cost = round(tokens * pricing.get(model_key, 0), 6)

    return {
        "latency_ms": round(latency_ms, 2),
        "tokens": tokens,
        "tokens_per_second": tps,
        "response_chars": len(response),
        "estimated_cost_usd": cost
    }

def calculate_latency_percentiles(latencies: list[float]):
    latencies.sort()

    return {
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(
            statistics.quantiles(latencies, n=100)[94],
            2
        )
    }