import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from utils.loader import load_txt_documents

# 🔐 OpenAI API key doğrudan burada
client = OpenAI(api_key="")

# 🧹 Temizlenecek ifadeler
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

# 📄 Belgeleri yükle
documents = load_txt_documents("data/documents")
print(f"✅ {len(documents)} dosya bulundu.")

# 🧼 Temizle (30 karakter sınırı yok!)
cleaned_docs = []
for doc in documents:
    for phrase in CLEANUP_PHRASES:
        doc = doc.replace(phrase, "")
    doc = doc.strip()
    cleaned_docs.append(doc)

# 💾 Temizlenmiş metinleri kaydet
os.makedirs("data/embeddings", exist_ok=True)
with open("data/embeddings/texts.pkl", "wb") as f:
    pickle.dump(cleaned_docs, f)

# 🔎 Embedding hesapla
def get_embeddings(texts):
    vectors = []
    for text in texts:
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append(response.data[0].embedding)
    return np.array(vectors, dtype=np.float32)

vectors = get_embeddings(cleaned_docs)

# 🧠 FAISS index oluştur ve kaydet
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, "data/embeddings/index.faiss")

print("✅ FAISS index ve temizlenmiş tüm metinler kaydedildi.")