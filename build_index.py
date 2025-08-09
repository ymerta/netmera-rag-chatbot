"""
Embedding Preparation Script for NetmerianBot

This script loads cleaned Netmera documentation and FAQ entries, generates OpenAI embeddings for each document,
and builds a FAISS index for efficient semantic search. The result is saved in two files:
- data/embeddings/texts.pkl: contains cleaned document texts with metadata (source, URL)
- data/embeddings/index.faiss: FAISS index storing the vector embeddings

Steps:
1. Loads all .txt documents from the data/documents directory.
2. Cleans up unwanted boilerplate phrases.
3. Appends formatted FAQ entries from data/faq_answers.json.
4. Embeds all text using OpenAI's embedding model.
5. Saves FAISS index and metadata for use in the chatbot.

Run this script every time you update the documentation files.
"""

import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from utils.loader import load_txt_documents 
from dotenv import load_dotenv
import json

load_dotenv()
with open("data/faq_answers.json", "r", encoding="utf-8") as f:
    faq_qa_map = json.load(f)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLEANUP_PHRASES = [
    "Was this helpful?",
    "Last updated",
    "Previous",
    "Next",
    "Copy",
    "Netmera Docs",
    "⌘",
    "K"
]

documents = load_txt_documents("data/documents")
print(f"{len(documents)} dosya bulundu.")

cleaned_docs = []
    """
    Removes boilerplate text from the input documents.

    Args:
        raw_documents (List[dict]): List of raw documents with 'text', 'source', and 'url'.
        cleanup_phrases (List[str]): List of phrases to remove from the text.

    Returns:
        List[dict]: List of cleaned documents with the same metadata.
    """
for doc in documents:
    text = doc["text"]
    for phrase in CLEANUP_PHRASES:
        text = text.replace(phrase, "")
    text = text.strip()
    cleaned_docs.append({
        "text": text,
        "source": doc["source"],
        "url": doc["url"]  
    })

for key, value in faq_qa_map.items():
    question = value["question"]
    answer = value["answer"]

    faq_entry = {
        "text": f"Q: {question}\nA: {answer}",
        "source": f"faq-{key}",
    }
    cleaned_docs.append(faq_entry)
    
os.makedirs("data/embeddings", exist_ok=True)
with open("data/embeddings/texts.pkl", "wb") as f:
    pickle.dump(cleaned_docs, f)

def get_embeddings(texts):
    """
    Generates OpenAI embeddings for a list of input texts.

    Args:
        texts (List[str]): A list of cleaned text documents or FAQ entries.

    Returns:
        np.ndarray: A NumPy array of shape (N, D) where N is the number of documents and D is the embedding dimension.
    """
    vectors = []
    for text in texts:
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append(response.data[0].embedding)
    return np.array(vectors, dtype=np.float32)

vectors = get_embeddings([d["text"] for d in cleaned_docs])

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, "data/embeddings/index.faiss")

print("FAISS index ve kaynaklı metinler başarıyla kaydedildi.")