from backend.app.llms.llama_local import LocalLlamaLLM
from backend.app.llms.mistral_local import LocalMistralLLM
from backend.app.llms.openai_llm import OpenAILLM
from backend.app.rag.retriever import retrieve
from backend.app.services.metrics import calculate_metrics
import time

llama = LocalLlamaLLM()
mistral = LocalMistralLLM()
gpt = OpenAILLM()


def run_llm_comparison(question: str):

    # ------------------------
    # RAG retrieval
    # ------------------------
    context_chunks = retrieve(question)
    context = "\n".join(context_chunks)

    prompt = f"""
You are an assistant answering questions using the following documentation.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    results = {}

    # ========================
    # Llama
    # ========================
    start = time.perf_counter()
    llama_response = llama.generate(prompt)
    end = time.perf_counter()

    llama_latency = (end - start) * 1000

    results["Llama 3"] = {
        "response": llama_response,
        "metrics": calculate_metrics(
            response=llama_response,
            latency_ms=llama_latency,
            model_key="llama"
        )
    }

    # ========================
    # Mistral
    # ========================
    start = time.perf_counter()
    mistral_response = mistral.generate(prompt)
    end = time.perf_counter()

    mistral_latency = (end - start) * 1000

    results["Mistral"] = {
        "response": mistral_response,
        "metrics": calculate_metrics(
            response=mistral_response,
            latency_ms=mistral_latency,
            model_key="mistral"
        )
    }

    # ========================
    # GPT
    # ========================
    start = time.perf_counter()
    gpt_response = gpt.generate(prompt)
    end = time.perf_counter()

    gpt_latency = (end - start) * 1000

    results["GPT"] = {
        "response": gpt_response,
        "metrics": calculate_metrics(
            response=gpt_response,
            latency_ms=gpt_latency,
            model_key="gpt"
        )
    }

    return results