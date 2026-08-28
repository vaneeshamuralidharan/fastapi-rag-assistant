# FastAPI Docs RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about FastAPI using its official documentation as a knowledge base — built end-to-end from scratch to understand each stage of a RAG system.

## What it does

1. **Scrapes** FastAPI's official tutorial docs and converts HTML to clean Markdown (`html2text`)
2. **Chunks** the text into overlapping segments (`langchain-text-splitters`)
3. **Embeds** each chunk into a vector using a local sentence-transformer model (`all-MiniLM-L6-v2`)
4. **Stores** embeddings in a local vector database (`ChromaDB`)
5. **Retrieves** the most relevant chunks for a user's question via semantic search
6. **Generates** an answer using a local, free LLM (`google/flan-t5-base`) grounded in retrieved context

## Example

**Question:** How do I declare a path parameter with a type?

**Answer:** URL = /files/file_path:path

## Setup

\`\`\`bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

cd src
python fetch_docs.py         # scrape + clean docs
python embed_docs.py         # chunk + embed + store
python generate_answer.py    # ask a question, get an answer
\`\`\`

## Architecture decisions & what I learned debugging this

- **Chunking:** Initial HTML-to-text extraction flattened all formatting. Switched to `html2text` to preserve paragraph and code-block structure, verified by inspecting individual chunks directly.
- **Retrieval:** Diagnosed a case where the correct chunk existed cleanly in the database but didn't rank in the top results, traced to the limitations of a small local embedding model rather than a chunking problem.
- **Generation:** Fixed three distinct failure modes: (1) context silently truncated before the question, producing garbage answers, (2) repetition loops under greedy decoding, (3) word hallucination from over-aggressive repetition penalties. Solved with beam search and tuned decoding parameters.

## Known limitations

- Uses a small local model (`flan-t5-base`) for zero-cost operation — answers are correct but terse compared to larger models
- Retrieval ranking isn't always precise for very specific technical queries, a known tradeoff of small embedding models

## Stack

Python · LangChain (text splitting) · Sentence-Transformers · ChromaDB · Hugging Face Transformers · BeautifulSoup · html2text