from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb

# --- Step 1: Load models ---
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading generation model (this downloads ~250MB the first time)...")
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# --- Step 2: Connect to ChromaDB ---
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="fastapi_docs")

# --- Step 3: The question ---
query = "How do I declare a path parameter with a type?"

# --- Step 4: Retrieve relevant chunks ---
query_embedding = embed_model.encode([query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=6)
retrieved_chunks = results["documents"][0]

# --- Step 5: Build the prompt ---
context = "\n\n".join(retrieved_chunks)
context=context[:3000]
prompt = f"""Question: {query}

Answer the question using only the context below.

Context:
{context}

Answer:"""
print("\n--- FULL PROMPT SENT TO MODEL ---")
print(prompt)
print("--- END PROMPT ---\n")
print("\n--- Retrieved context (first 300 chars) ---")
print(context[:300], "...\n")

# --- Step 6: Generate the answer ---
print("Generating answer...")
inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
output_ids = gen_model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=4,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
    early_stopping=True,
)
answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n--- Final Answer ---")
print(answer)