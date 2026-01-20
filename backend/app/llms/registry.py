from .llama_local import LocalLlamaLLM
from .mistral_local import LocalMistralLLM

LLM_REGISTRY = {
    "llama3": LocalLlamaLLM(),
    "mistral": LocalMistralLLM()
}