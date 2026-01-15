from fastapi import FastAPI

app = FastAPI(title="LLM Comparison RAG POC")

@app.get("/health")
def health():
    return {"status": "ok"}