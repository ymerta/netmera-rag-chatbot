import os
import shutil
import faiss
import pickle
import numpy as np
from openai import OpenAI

# 🔐 OpenAI API key doğrudan burada
client = OpenAI(api_key="")

# 📂 Dosya yolları
index_path = "data/embeddings/index.faiss"
texts_path = "data/embeddings/texts.pkl"
new_docs_dir = "data/new_docs"
existing_docs_dir = "data/documents"

# 🧹 Temizlenecek ifadeler
CLEANUP_PHRASES = [
    "Last updated", "Was this helpful?", "Next", "Previous", "Copy", "Netmera Docs",
    "BEGINNER'S GUIDE TO NETMERA", "⌘", "K"
]

# 🔄 FAISS index ve metadata yükle
index = faiss.read_index(index_path)
with open(texts_path, "rb") as f:
    metadata = pickle.load(f)

# 🧠 Eski format dönüştür
if isinstance(metadata[0], str):
    metadata = [{"text": para, "source": f"legacy_{i}"} for i, para in enumerate(metadata)]

# ✅ Zaten işlenmiş dosyalar
processed_files = set(item["source"] for item in metadata)
existing_filenames = set(f.replace(".txt", "") for f in os.listdir(existing_docs_dir) if f.endswith(".txt"))

# 📑 Embedlenecek dosyalar
new_files = [
    f for f in os.listdir(new_docs_dir)
    if f.endswith(".txt") and f.replace(".txt", "") not in processed_files and f.replace(".txt", "") not in existing_filenames
]

if not new_files:
    print("🚫 Yeni dosya bulunamadı veya zaten işlenmiş.")
    exit()

# 📥 İşleme başla
for filename in new_files:
    filepath = os.path.join(new_docs_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ✂️ Gereksiz kelimeleri temizle
    for phrase in CLEANUP_PHRASES:
        content = content.replace(phrase, "")

    # 🔄 Boş olmayan tüm paragrafları al (uzunluk filtresi yok)
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

    for para in paragraphs:
        response = client.embeddings.create(
            input=[para],
            model="text-embedding-ada-002"
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

        index.add(embedding)
        metadata.append({
            "text": para,
            "source": filename.replace(".txt", "")
        })

    # ✅ Dosyayı taşı
    shutil.move(filepath, os.path.join(existing_docs_dir, filename))

# 💾 Kaydet
faiss.write_index(index, index_path)
with open(texts_path, "wb") as f:
    pickle.dump(metadata, f)

print(f"✅ Yeni dosyalar işlendi ve taşındı: {len(new_files)} adet")