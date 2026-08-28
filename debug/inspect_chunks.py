from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

data_folder = "data"

with open(os.path.join(data_folder, "path_parameters.txt"), "r", encoding="utf-8") as f:
    text = f.read()

chunks = splitter.split_text(text)

print(f"path_parameters.txt split into {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks):
    if "item_id" in chunk and ("int" in chunk):
        print(f"=== Chunk {i} (length {len(chunk)}) ===")
        print(chunk)
        print()