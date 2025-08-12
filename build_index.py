"""
Build embeddings & FAISS index for NetmerianBot (User + Developer Guide + FAQ)
Outputs:
- data/embeddings/texts.pkl   # [{id, text, source, url}]
- data/embeddings/index.faiss # FAISS IndexFlatIP (cosine)
"""

import os, re, json, pickle, math, time
from typing import List, Dict
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
os.makedirs("data/embeddings", exist_ok=True)

DOC_DIR = "data/documents"
FAQ_PATH = "data/faq_answers.json"
TEXTS_OUT = "data/embeddings/texts.pkl"
INDEX_OUT = "data/embeddings/index.faiss"

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")  # istersen text-embedding-3-large
BATCH_SIZE = 64
MAX_CHARS = 4500         # chunk hedefi ~800-1000 token (yaklaşık)
OVERLAP_CHARS = 800

BASE_DOC_URL_USER = "https://user.netmera.com/netmera-user-guide"
BASE_DOC_URL_DEV  = "https://user.netmera.com/netmera-developer-guide"
FAQ_URL = "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/faqs"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_txt_with_url(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    m = re.match(r"^\[SOURCE_URL\]:\s*(\S+)\s*\n", raw)
    url = m.group(1) if m else None
    text = raw[m.end():] if m else raw
    return {"text": text.strip(), "url": url}

def filename_to_url(filename: str) -> str:
    # Fallback: dosya adına göre URL üret
    name = filename
    if name.endswith(".txt"): name = name[:-4]
    if name.startswith("netmera-user-guide-"):
        path = name[len("netmera-user-guide-"):].replace("-", "/")
        return f"{BASE_DOC_URL_USER}/{path}"
    if name.startswith("netmera-developer-guide-"):
        path = name[len("netmera-developer-guide-"):].replace("-", "/")
        return f"{BASE_DOC_URL_DEV}/{path}"
    return BASE_DOC_URL_USER

def sentence_split(s: str) -> List[str]:
    # Basit cümle kesici (nokta, soru, ünlem, yeni satır)
    parts = re.split(r"(?<=[\.!?])\s+|\n+", s)
    return [p.strip() for p in parts if p and len(p.strip()) > 0]

def chunk_text(s: str, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS) -> List[str]:
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if len(s) <= max_chars:
        return [s]
    sent = sentence_split(s)
    chunks, cur = [], ""
    for t in sent:
        if len(cur) + 1 + len(t) <= max_chars:
            cur = (cur + " " + t) if cur else t
        else:
            if cur: chunks.append(cur)
            # overlap: son kısmı kesip yeni chunk’a taşı
            if overlap > 0 and len(cur) > overlap:
                cur = cur[-overlap:] + " " + t
            else:
                cur = t
    if cur: chunks.append(cur)
    return chunks

def load_docs() -> List[Dict]:
    docs = []
    for fn in sorted(os.listdir(DOC_DIR)):
        if not fn.endswith(".txt"): continue
        meta = read_txt_with_url(os.path.join(DOC_DIR, fn))
        url = meta["url"] or filename_to_url(fn)
        text = meta["text"]
        # chunk’la
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            docs.append({
                "id": f"{fn}::chunk{i}",
                "text": ch,
                "source": fn,
                "url": url
            })
    # FAQ ekle
    if os.path.exists(FAQ_PATH):
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            faq = json.load(f)
        for key, v in faq.items():
            q, a = v.get("question","").strip(), v.get("answer","").strip()
            if not q or not a: continue
            docs.append({
                "id": f"faq::{key}",
                "text": f"Q: {q}\nA: {a}",
                "source": f"faq-{key}",
                "url": FAQ_URL
            })
    return docs

def embed_batch(texts: List[str]) -> np.ndarray:
    # OpenAI embeddings batch
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        for retry in range(3):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                for item in resp.data:
                    out.append(item.embedding)
                break
            except Exception as e:
                if retry == 2:
                    raise
                time.sleep(1.5 * (retry + 1))
    arr = np.array(out, dtype=np.float32)
    return arr

def build_faiss(vectors: np.ndarray) -> faiss.Index:
    # cosine için normalize + inner product index
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def main():
    docs = load_docs()
    print(f"Loaded {len(docs)} chunks (docs+faq). Embedding...")
    vectors = embed_batch([d["text"] for d in docs])
    print("Vectors:", vectors.shape)
    index = build_faiss(vectors)

    # persist
    with open(TEXTS_OUT, "wb") as f:
        pickle.dump(docs, f)
    faiss.write_index(index, INDEX_OUT)
    print("Saved:", TEXTS_OUT, "and", INDEX_OUT)

if __name__ == "__main__":
    main()