import streamlit as st
import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from langdetect import detect
from dotenv import load_dotenv
load_dotenv()

# 🔐 API anahtarı doğrudan girildi
client = OpenAI(api_key=st.secrets("OPENAI_API_KEY"))

# 📦 FAISS index ve metinler
index = faiss.read_index("data/embeddings/index.faiss")
with open("data/embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# 🔎 Embed
def embed_question(question):
    response = client.embeddings.create(
        input=[question],
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
# 🌐 Dili otomatik algıla
def detect_language(text):
    try:
        lang_code = detect(text)
        if lang_code == "tr":
            return "Türkçe"
        else:
            return "English"
    except:
        return "English"  # Default

# 🤖 Yanıt üret
def ask_openai(question, context, lang="English"):
    if lang == "Türkçe":
        prompt = f"""
Sen Netmera platformunda çalışan kıdemli bir destek mühendisisin.
Aşağıdaki bilgileri kullanarak kullanıcının sorusunu yanıtla.

DOKÜMAN:
{context}

SORU:
{question}

YANIT:"""
    else:
        prompt = f"""
You are a senior support engineer for the Netmera platform.
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

# 🎨 Arayüz

# 🌐 Dil seçimi
lang_manual = st.toggle("🌐 Dili manuel seç", value=False)
if lang_manual:
    lang = st.radio("Dil / Language", ("Türkçe", "English"), horizontal=True)
else:
    lang = None  # Otomatik olarak belirlenecek

st.set_page_config(page_title="NetmerianBot", layout="centered")
st.title("🤖 NetmerianBot")

# 💾 Sohbet geçmişi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📌 Sık Sorular Butonları
if lang == "Türkçe":
    st.markdown("### 📌 Sık Sorulan Konular")
    faq_questions = [
        "Push gönderiminde 'Send All' özelliği tüm kullanıcılara ulaşır mı?",
        "Toplu mesaj gönderimi yarıda durdurulabilir mi?",
        "Netmera SDK hangi kullanıcı davranışlarını takip eder?",
        "Push gönderim başarısız olduğunda sistem uyarı verir mi?",
        "Segmentler ne zaman pasif hale gelir?",
        "Push gönderiminde buton eklemek API ile mümkün mü?",
        "Netmera, kullanıcı uygulamayı silince bunu nasıl anlar?",
        "Funnel verileri neden değişiklik gösterir?",
        "Netmera web'de anonim kullanıcıları nasıl izler?",
    ]
    input_placeholder = "Bir soru yazın..."
    no_info_message = "⚠️ Bu konuda yeterli bilgi yok. Lütfen daha açık şekilde sorun."
else:
    st.markdown("### 📌 Frequently Asked Questions")
    faq_questions = [
        "Does 'Send All' option deliver push to all users?",
        "Can I stop bulk push sending midway?",
        "Which user behaviors does Netmera SDK track?",
        "Does Netmera warn if a push fails?",
        "When do segments become inactive?",
        "Can buttons be added to push via API?",
        "How does Netmera detect when a user uninstalls the app?",
        "Why do funnel values fluctuate?",
        "How does Netmera track anonymous web users?",
    ]
    input_placeholder = "Type a question..."
    no_info_message = "⚠️ There is not enough information on this topic. Please ask more clearly."

cols = st.columns(2)
selected_question = None
for i, q in enumerate(faq_questions):
    if cols[i % 2].button(q, key=f"faq_{i}"):
        selected_question = q

# 👤 Manuel yazım
user_input = st.chat_input(input_placeholder)

# Eğer butondan seçim varsa onu kullan
if selected_question and not user_input:
    user_input = selected_question

# 🔁 RAG süreci – son mesajla aynıysa tekrar ekleme
if user_input and (len(st.session_state.chat_history) == 0 or user_input != st.session_state.chat_history[-1][1]):
    
    # 🌐 Eğer manuel seçim yapılmadıysa, dili otomatik tespit et
    if not lang:
        lang = detect_language(user_input)
        st.info(f"🧠 Algılanan dil: {lang}")
    
    st.session_state.chat_history.append(("user", user_input))

    embedding = embed_question(user_input)
    distances, indices = index.search(embedding, k=1)
    score = distances[0][0]
    selected_doc = texts[indices[0][0]]
    paragraph = selected_doc["text"]
    source_file = selected_doc["source"]

    if score > 0.6:
        answer = no_info_message
    else:
        answer_text = ask_openai(user_input, paragraph, lang)
        answer = f"{answer_text}\n\n📄 **Kaynak belge**: `{source_file}`"
        
    st.session_state.chat_history.append(("assistant", answer))

    

# 💬 Geçmişi göster
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)