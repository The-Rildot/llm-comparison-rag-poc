import os
from openai import OpenAI
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
You are an AI assistant answering questions about an internal system.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
