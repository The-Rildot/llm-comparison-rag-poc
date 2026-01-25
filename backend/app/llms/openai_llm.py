import os
from dotenv import load_dotenv
from openai import OpenAI
from .base import BaseLLM

load_dotenv()

SYSTEM_PROMPT = """
You are an enterprise AI assistant.

Answer ONLY using the information provided in the context below.

If the answer is not contained in the context, respond exactly with:

"I do not have enough information to answer this question."

Do not use outside knowledge.
Do not guess.
Do not hallucinate.
"""

class OpenAILLM(BaseLLM):

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, context: str = "", stream: bool = False):

        prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""

        # 🔹 NON-STREAMING
        if not stream:
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        # 🔹 STREAMING
        def token_generator():
            stream_response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            for event in stream_response:
                if event.choices:
                    delta = event.choices[0].delta
                    if delta and delta.content:
                        yield delta.content

        return token_generator()
