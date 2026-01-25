import time
import statistics


def safe_generate(model, question, context, stream=False):
    try:
        return model.generate(
            question=question,
            context=context,
            stream=stream
        )
    except Exception as e:
        return f"[ERROR] {str(e)}"


def run_model(model, question, context, stream=False):
    start = time.perf_counter()

    result = safe_generate(
        model,
        question,
        context,
        stream=stream
    )

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    return {
        "response": result,
        "latency_ms": round(latency_ms, 2)
    }


def run_model_multiple_times(
    model,
    question,
    context,
    runs=5
):
    latencies = []
    final_response = None

    for _ in range(runs):
        start = time.perf_counter()
        response = safe_generate(model, question, context, stream=False)
        end = time.perf_counter()

        latency = (end - start) * 1000
        latencies.append(latency)
        final_response = response

    return {
        "response": final_response,
        "metrics": {
            "runs": runs,
            "latency_avg_ms": round(statistics.mean(latencies), 2),
            "latency_p50_ms": round(statistics.median(latencies), 2),
            "latency_p95_ms": round(statistics.quantiles(latencies, n=20)[18], 2),
        }
    }