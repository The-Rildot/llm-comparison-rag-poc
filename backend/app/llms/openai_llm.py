import os
from dotenv import load_dotenv
from openai import OpenAI
from .base import BaseLLM

load_dotenv()

SYSTEM_PROMPT = """
You are an enterprise AI assistant.
Answer using ONLY the context provided.
If the answer is not in the context, say "I do not have enough information."
"""

class OpenAILLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, context: str = "") -> str:
        prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""
        response = self.client.chat.completions.create(
            model="gpt-5-nano",  # cheaper model
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content