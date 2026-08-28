from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

data_folder = "../data"
all_chunks = []

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        print(f"{filename}: split into {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append({"source": filename, "text": chunk})

print(f"\nTotal chunks across all files: {len(all_chunks)}")

# Print the first chunk from the first file so we can inspect it
print("\n--- Sample chunk ---")
print(all_chunks[0]["text"])