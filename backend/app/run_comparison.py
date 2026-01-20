from rag.retriever import retrieve
from llms.registry import LLM_REGISTRY

question = "What can the admin do that the user cannot?"

docs = retrieve(question)
context = "\n\n".join(docs)

for name, llm in LLM_REGISTRY.items():
    print(f"\n--- {name.upper()} ---")
    print(llm.generate(question, context))
