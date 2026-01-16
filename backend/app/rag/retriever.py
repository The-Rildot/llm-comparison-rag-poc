from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load the same embedding model used during ingestion
model = SentenceTransformer("all-MiniLM-L6-v2")

# Resolve project root directory
BASE_DIR = Path.cwd()

# Persistent Chroma storage location (MUST match ingest.py)
CHROMA_PATH = BASE_DIR / "backend" / "app" / "rag" / "chroma_db"

# Initialize persistent Chroma client
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Collection name MUST match ingest.py
collection = client.get_or_create_collection("enterprise_docs")

# Function to retrieve top-k relevant document chunks for a given query
# Works by embedding the query and comparing it to stored embeddings to find closest matches
def retrieve(query: str, k: int = 3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    print("CHROMA PATH:", CHROMA_PATH)

    return results["documents"][0]