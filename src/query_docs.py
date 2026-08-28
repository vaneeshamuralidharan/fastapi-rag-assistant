from sentence_transformers import SentenceTransformer
import chromadb

# Load the same embedding model used before — must match, or vectors won't compare meaningfully
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="fastapi_docs")

# The question we want to ask
query = "How do I declare a path parameter with a type?"

# Embed the query the same way we embedded the chunks
query_embedding = model.encode([query]).tolist()

# Ask ChromaDB for the most similar chunks
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
)

print(f"Query: {query}\n")
for i, (doc, meta, distance) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
)):
    print(f"--- Result {i+1} (source: {meta['source']}, distance: {distance:.4f}) ---")
    print(doc)
    print()