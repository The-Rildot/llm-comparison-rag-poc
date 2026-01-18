from rag.retriever import retrieve
from llms.llama_local import LocalLlamaLLM

llm = LocalLlamaLLM()

question = "Can assets be deleted?"

docs = retrieve(question)
context = "\n\n".join(docs)

answer = llm.generate(question, context)

print("\nANSWER:\n")
print(answer)
