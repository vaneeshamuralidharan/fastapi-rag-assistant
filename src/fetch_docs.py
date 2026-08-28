import requests
from bs4 import BeautifulSoup
import html2text
import os

urls = {
    "first_steps": "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "path_parameters": "https://fastapi.tiangolo.com/tutorial/path-params/",
    "query_parameters": "https://fastapi.tiangolo.com/tutorial/query-params/",
    "request_body": "https://fastapi.tiangolo.com/tutorial/body/",
}

converter = html2text.HTML2Text()
converter.ignore_links = True   # we don't need hyperlinks for embeddings
converter.body_width = 0        # don't wrap lines, keep source structure clean

os.makedirs("../data", exist_ok=True)

for name, url in urls.items():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article")

    markdown_text = converter.handle(str(article))

    filepath = f"../data/{name}.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Saved {filepath} ({len(markdown_text)} characters)")