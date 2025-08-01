import streamlit as st
import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from langdetect import detect
from dotenv import load_dotenv
import csv
from datetime import datetime
import pandas as pd
import json
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

from config import (
    EMBEDDING_MODEL, CHAT_MODEL,
    BM25_WEIGHT, FAISS_WEIGHT, FUZZY_WEIGHT,
    EMBEDDINGS_PATH, TEXTS_PATH, FAQ_PATH,
    BASE_DOC_URL, FAQ_URL,
    SYSTEM_PROMPT, TRANSLATE_PROMPT, TURKISH_TRANSLATION_PROMPT,
    FAQ_QUESTIONS_TR, FAQ_QUESTIONS_EN
)
nltk.download('all')

load_dotenv()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

index = faiss.read_index(EMBEDDINGS_PATH)
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)
    
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_qa_map = json.load(f)

corpus = [doc["text"] for doc in texts]
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
bm25_model = BM25Okapi(tokenized_corpus)

def compute_hybrid_score(doc, norm_bm25, norm_faiss, norm_fuzzy):
    return BM25_WEIGHT * norm_bm25 + FUZZY_WEIGHT * norm_fuzzy + FAISS_WEIGHT * norm_faiss

def check_faq_match(translated_input, threshold=80):
    best_score = 0
    best_answer = None
    best_source = None

    for key, entry in faq_qa_map.items():
        score = fuzz.partial_ratio(translated_input.lower(), key.replace("_", " ").lower())
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]
            best_source = entry["source"]

    if best_score >= threshold:
        return f"{best_answer}\n\n **Kaynak belge**: [FAQ]({best_source})"
    return None

def detect_language(text):
    try:
        lang_code = detect(text)
        return "Türkçe" if lang_code == "tr" else "English"
    except:
        return "English"  

def filename_to_url(filename: str) -> str:
    if filename.startswith("faq-"):
        return FAQ_URL
    if filename.startswith("netmera-user-guide-"):
        filename = filename[len("netmera-user-guide-"):]
    if filename.endswith(".txt"):
        filename = filename[:-4]
    url_path = filename.replace("-", "/")
    return f"{BASE_DOC_URL}/{url_path}"

def embed_question(translated_input):
    response_embed = client.embeddings.create(
        input=[translated_input], model=EMBEDDING_MODEL
    )
    return np.array(response_embed.data[0].embedding, dtype=np.float32).reshape(1, -1)

def ask_openai(question, context, lang="English"):
    user_prompt = f"""
CONTENT:
{context}

QUESTION:
{question}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    english_answer = response.choices[0].message.content.strip()

    if lang == "Türkçe":
        translation = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": TURKISH_TRANSLATION_PROMPT},
                {"role": "user", "content": english_answer},
            ],
        )
        turkish_answer = translation.choices[0].message.content.strip()
        return turkish_answer if len(turkish_answer) >= 5 else english_answer
    return english_answer

lang_manual = st.toggle("Dil seç", value=False)
lang = st.radio("Dil / Language", ("Türkçe", "English"), horizontal=True) if lang_manual else None

st.set_page_config(page_title="NetmerianBot", layout="centered")
st.title("🤖 NetmerianBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

faq_questions = FAQ_QUESTIONS_TR if lang == "Türkçe" else FAQ_QUESTIONS_EN
input_placeholder = "Bir soru yazın..." if lang == "Türkçe" else "Type a question..."
no_info_message = "⚠️ Bu konuda yeterli bilgi yok. Lütfen daha açık şekilde sorun." if lang == "Türkçe" else "⚠️ There is not enough information on this topic. Please ask more clearly."

cols = st.columns(2)
selected_question = None
for i, q in enumerate(faq_questions):
    if cols[i % 2].button(q, key=f"faq_{i}"):
        selected_question = q

user_input = st.chat_input(input_placeholder)
if selected_question and not user_input:
    user_input = selected_question

if user_input and (len(st.session_state.chat_history) == 0 or user_input != st.session_state.chat_history[-1][1]):
    try:
        translation = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        translated_input = translation.choices[0].message.content.strip()
    except Exception:
        translated_input = user_input

    faq_response = check_faq_match(translated_input)
    st.session_state.chat_history.append(("user", user_input))

    if faq_response:
        st.session_state.chat_history.append(("assistant", faq_response))
    else:
        if not lang:
            lang = detect_language(user_input)

        embedding = embed_question(translated_input)
        embedding_vector = embedding[0]
        tokenized_query = word_tokenize(user_input.lower())
        bm25_scores = bm25_model.get_scores(tokenized_query)
        bm25_mean = np.mean(bm25_scores)
        bm25_std = np.std(bm25_scores) or 1.0

        candidate_docs = []
        for idx, doc in enumerate(texts):
            doc_embedding = index.reconstruct(idx)
            norm_doc = np.linalg.norm(doc_embedding)
            norm_query = np.linalg.norm(embedding_vector)
            cosine_sim = np.dot(doc_embedding, embedding_vector) / (norm_doc * norm_query) if norm_doc and norm_query else 0.0
            bm25_score = bm25_scores[idx]
            norm_bm25 = (bm25_score - bm25_mean) / bm25_std
            fuzzy_score = fuzz.partial_ratio(translated_input.lower(), doc["text"][:1000].lower()) / 100
            hybrid = compute_hybrid_score(doc, norm_bm25, cosine_sim, fuzzy_score)

            doc.update({
                "faiss_score": cosine_sim,
                "bm25_score": norm_bm25,
                "fuzzy_score": fuzzy_score,
                "hybrid_score": hybrid
            })
            candidate_docs.append(doc)

        best_doc = max(candidate_docs, key=lambda d: d["hybrid_score"])
        top_docs = sorted(candidate_docs, key=lambda d: d["hybrid_score"], reverse=True)[:3]
        top_k_context = "\n\n---\n\n".join([doc["text"] for doc in top_docs])
        answer_text = ask_openai(user_input, top_k_context, lang)
        source_file = best_doc["source"]
        source_url = best_doc.get("url") or filename_to_url(source_file)
        short_name = source_url.replace("https://user.netmera.com/netmera-user-guide/", "").replace("-", " ").replace("/", " > ").title()
        label_source = "📄 **Kaynak belge**" if lang == "Türkçe" else "📄 **Source document**"
        answer = f"{answer_text}\n\n{label_source}: [{short_name}]({source_url})"
        st.session_state.chat_history.append(("assistant", answer))

for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)