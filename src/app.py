import os

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Works no matter which folder you launch it from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

st.set_page_config(page_title="FastAPI Docs Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading models (the first run takes a minute)...")
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("fastapi_docs")
    return embedder, tokenizer, model, collection


def retrieve(question, embedder, collection, k):
    embedding = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=embedding, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def generate(question, chunks, tokenizer, model):
    context = "\n\n".join(c[0] for c in chunks)[:3000]
    prompt = f"""Question: {question}

Answer the question using only the context below.

Context:
{context}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------- Interface ----------
st.title("FastAPI Docs Assistant")
st.caption("Retrieval-Augmented Generation over the FastAPI tutorial. All models run locally.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Your question is converted to a vector\n"
        "2. ChromaDB returns the closest passages\n"
        "3. The language model writes an answer from them"
    )
    k = st.slider("Passages to retrieve", 1, 10, 6)
    show_sources = st.checkbox("Show retrieved passages", value=True)

examples = [
    "How do I declare a path parameter with a type?",
    "How do I make a query parameter optional?",
    "How are path, query and body parameters told apart?",
    "What error appears if I send a string where an int is expected?",
]
choice = st.selectbox("Pick an example question, or type your own below", [""] + examples)
question = st.text_input("Your question", value=choice)

if st.button("Ask", type="primary") and question.strip():
    embedder, tokenizer, model, collection = load_resources()
    with st.spinner("Retrieving passages..."):
        chunks = retrieve(question, embedder, collection, k)
    with st.spinner("Generating answer..."):
        answer = generate(question, chunks, tokenizer, model)
    st.session_state["result"] = (question, answer, chunks)

if "result" in st.session_state:
    q, answer, chunks = st.session_state["result"]
    st.subheader("Answer")
    st.caption(f"Question: {q}")
    st.success(answer if answer.strip() else "The model returned no answer for this question.")
    if show_sources:
        st.subheader("Retrieved passages")
        st.caption("Lower distance means a closer match.")
        for i, (text, meta, dist) in enumerate(chunks, 1):
            with st.expander(f"{i}. {meta['source']}  |  distance {dist:.3f}", expanded=(i == 1)):
                st.text(text)