from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

# --- Step 1: Chunk the documents (same as before) ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

data_folder = "../data"
all_chunks = []
all_sources = []

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(filename)

print(f"Total chunks to embed: {len(all_chunks)}")

# --- Step 2: Load the embedding model ---
print("Loading embedding model (this may take a moment the first time)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 3: Generate embeddings ---
print("Generating embeddings...")
embeddings = model.encode(all_chunks).tolist()
print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")

# --- Step 4: Store in ChromaDB ---
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_or_create_collection(name="fastapi_docs")

ids = [f"chunk_{i}" for i in range(len(all_chunks))]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=all_chunks,
    metadatas=[{"source": s} for s in all_sources],
)

print(f"\nStored {collection.count()} chunks in ChromaDB at ../chroma_db")