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