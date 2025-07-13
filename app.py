import streamlit as st
import os
import faiss
import pickle
import numpy as np
import openai
from dotenv import load_dotenv
from openai import OpenAI

# API anahtarını .env dosyasından al
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# FAISS index ve ilgili döküman metinlerini yükle (Retrieval için kullanılan vektör veritabanı)
index = faiss.read_index("data/embeddings/index.faiss")
with open("data/embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)

#Soruyu embed’leyerek vektör haline getir (Retrieval aşaması – 1. adım)
def embed_question(question):
    response = client.embeddings.create(
        input=[question],
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

# En alakalı paragrafı GPT’ye verip cevap üret (Generation aşaması – 2. adım)
def ask_openai(question, context):
    prompt = f"""
You are a helpful assistant for the Netmera platform.
Use the information below to answer the user's question.

DOCUMENT:
{context}

QUESTION:
{question}

ANSWER:"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Streamlit arayüzünü başlat
st.set_page_config(page_title="Netmera Chatbot", layout="centered")
st.title("🤖 NetmerianBot")

# Sohbet geçmişini saklamak için session_state kullan
if "history" not in st.session_state:
    st.session_state.history = []

# Kullanıcıdan gelen soruyu al
user_input = st.text_input("Size nasıl yardımcı olabilirim?")

if st.button("Sor") and user_input:
    # RAG - Step 1: Soruyu embed’le, FAISS ile en yakın dökümanı bul
    embedding = embed_question(user_input)
    distances, indices = index.search(embedding, k=1)
    score = distances[0][0]
    paragraph = texts[indices[0][0]]

    # RAG - Step 2: En uygun dökümanı GPT'ye vererek yanıt oluştur
    if score > 0.6:
        answer = "⚠️ Bu konuda yeterli bilgi yok. Lütfen farklı şekilde sorun."
    else:
        answer = ask_openai(user_input, paragraph)

    # 💬 Sohbet geçmişine ekle
    st.session_state.history.append(("🧑‍💻 " + user_input, "🤖 " + answer))

# Önceki konuşmaları aşağıda göster
for user_msg, bot_msg in st.session_state.history[::-1]:
    st.markdown(f"**{user_msg}**")
    st.markdown(f"{bot_msg}")
    st.markdown("---")