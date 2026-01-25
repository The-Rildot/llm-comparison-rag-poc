import ollama
from .base import BaseLLM


SYSTEM_PROMPT = """
You are an enterprise AI assistant.

Answer ONLY using the information provided in the context below.

If the answer is not contained in the context, respond exactly with:

"I do not have enough information to answer this question."

Do not use outside knowledge.
Do not guess.
Do not hallucinate.
"""

class LocalLlamaLLM(BaseLLM):
    def __init__(self, model_name="llama3:8b"):
        self.model_name = model_name

    def generate(self, question: str, context: str = "", stream: bool = False):

        messages = [
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

        # 🔹 NON-STREAMING
        if not stream:
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            return response["message"]["content"]

        # 🔹 STREAMING
        def token_generator():
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True
            )

            for chunk in response:
                if "message" in chunk:
                    yield chunk["message"]["content"]

        return token_generator()