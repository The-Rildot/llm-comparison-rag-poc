from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configuration for chunking, 500 characters with 50 characters overlap
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Function to chunk text into overlapping segments
def chunk_text(text: str):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# Main ingestion function, reading markdown files, chunking, embedding, and storing in ChromaDB
def ingest():
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Resolve project root directory
    BASE_DIR = Path.cwd()

    # Persistent Chroma storage location
    CHROMA_PATH = BASE_DIR / "backend" / "app" / "rag" / "chroma_db"

    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    print("CHROMA PATH:", CHROMA_PATH)

    # Collection name MUST match retriever
    collection = client.get_or_create_collection("enterprise_docs")

    # Absolute data directory (important!)
    docs_path = BASE_DIR / "data"

    total_chunks = 0

    for file in docs_path.glob("*.md"):
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()

        ids = [f"{file.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file.name} for _ in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        total_chunks += len(chunks)

    print(f"Ingested {total_chunks} chunks")

# Run the ingestion process if this script is executed directly
if __name__ == "__main__":
    ingest()