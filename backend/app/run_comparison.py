from app.rag.retriever import retrieve
from app.rag.prompt import build_context
from app.llms.openai_llm import OpenAILLM

question = "Can assets be deleted?"

chunks = retrieve(question)
context = build_context(chunks)

llm = OpenAILLM()
answer = llm.generate(question, context)

print("QUESTION:", question)
print("\nCONTEXT:\n", context)
print("\nANSWER:\n", answer)