from .llama_local import LocalLlamaLLM
from .mistral_local import LocalMistralLLM
from .openai_llm import OpenAILLM

LLM_REGISTRY = {
    "llama3": LocalLlamaLLM(),
    "mistral": LocalMistralLLM(),
    "openai": OpenAILLM(),
}