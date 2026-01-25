from backend.app.llms.llama_local import LocalLlamaLLM
from backend.app.llms.mistral_local import LocalMistralLLM
from backend.app.llms.openai_llm import OpenAILLM
from backend.app.rag.retriever import retrieve

from backend.app.services.evaluation_harness import (
    run_model,
    run_model_multiple_times
)

llama = LocalLlamaLLM()
mistral = LocalMistralLLM()
gpt = OpenAILLM()


def run_llm_comparison(
    question: str,
    mode: str = "single",     # "single" | "multiple"
    stream: bool = False
):
    context_chunks = retrieve(question)
    context = "\n".join(context_chunks)

    models = {
        "Llama 3": llama,
        "Mistral": mistral,
        "GPT": gpt
    }

    results = {}

    for name, model in models.items():

        if mode == "multiple":
            results[name] = run_model_multiple_times(
                model=model,
                question=question,
                context=context,
                runs=5
            )

        else:
            results[name] = run_model(
                model=model,
                question=question,
                context=context,
                stream=stream
            )

    return results