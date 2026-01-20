import ollama
from .base import BaseLLM

SYSTEM_PROMPT = """
You are an enterprise AI assistant.

Answer ONLY using the context below.
If the answer is not present, say:
"I do not have enough information to answer this question."
"""

class LocalMistralLLM(BaseLLM):
    def generate(self, question: str, context: str) -> str:
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}
"""
                }
            ]
        )

        return response["message"]["content"]