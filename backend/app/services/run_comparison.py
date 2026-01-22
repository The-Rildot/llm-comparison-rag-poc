from backend.app.llms.llama_local import LocalLlamaLLM
from backend.app.llms.mistral_local import LocalMistralLLM
from backend.app.llms.openai_llm import OpenAILLM
from backend.app.rag.retriever import retrieve


llama = LocalLlamaLLM()
mistral = LocalMistralLLM()
gpt = OpenAILLM()


def run_llm_comparison(question: str):
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

    return {
        "Llama 3": llama.generate(prompt),
        "Mistral": mistral.generate(prompt),
        "GPT": gpt.generate(prompt)
    }