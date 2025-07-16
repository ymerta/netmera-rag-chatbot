import os

index_path = "data/embeddings/index.faiss"
texts_path = "data/embeddings/texts.pkl"

deleted = []

if os.path.exists(index_path):
    os.remove(index_path)
    deleted.append("✅ FAISS index silindi: index.faiss")

if os.path.exists(texts_path):
    os.remove(texts_path)
    deleted.append("✅ Metin verisi silindi: texts.pkl")

if not deleted:
    print("⚠️ Silinecek bir şey bulunamadı. Zaten temiz görünüyor.")
else:
    for msg in deleted:
        print(msg)

print("🔁 Sistem sıfırlandı. Şimdi build_index.py çalıştırabilirsiniz.")