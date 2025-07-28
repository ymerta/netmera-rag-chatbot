
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


load_dotenv()

# 🔐 API anahtarı doğrudan girildi
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 📦 FAISS index ve metinler
index = faiss.read_index("data/embeddings/index.faiss")
with open("data/embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)
    
# 📚 Netmera FAQ JSON dosyasını yükle
with open("data/faq_answers.json", "r", encoding="utf-8") as f:
    faq_qa_map = json.load(f)

corpus = [doc["text"] for doc in texts]
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
bm25_model = BM25Okapi(tokenized_corpus)

def compute_hybrid_score(doc, bm25_score, faiss_score, fuzzy_score):
    norm_bm25 = bm25_score / 100
    norm_faiss = -faiss_score  # çünkü FAISS uzaklık: daha küçük daha iyi
    norm_fuzzy = fuzzy_score / 100
    return 0.4 * norm_bm25 + 0.3 * norm_fuzzy + 0.3 * norm_faiss

# ✅ Kullanıcının sorusu sık sorulardan biriyle eşleşiyor mu?
def check_faq_match(user_input, threshold=80):
    lowered = user_input.lower()
    best_score = 0
    best_answer = None
    best_source = None

    for key, entry in faq_qa_map.items():
        score = fuzz.partial_ratio(lowered, key.replace("_", " ").lower())
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]
            best_source = entry["source"]

    if best_score >= threshold:
        return f"{best_answer}\n\n📄 **Kaynak belge**: [FAQ]({best_source})"
    return None

# 🔁 Belirli dosyalar için manuel URL eşlemeleri
file_to_url_map = {
    "netmera-user-guide-messages-e-mail-email-onboarding-iys-ileti-yoenetim-sistemi.txt": "https://user.netmera.com/netmera-user-guide/messages/email/email-onboarding/iys-ileti-yonetim-sistemi",
    "netmera-user-guide-customer-data-iys-integration.txt": "https://user.netmera.com/netmera-user-guide/customer-data/iys-integration",
    "netmera-user-guide-messages-multi-language-push.txt": "https://user.netmera.com/netmera-user-guide/messages/multi-language-push",
    "netmera-user-guide-beginners-guide-to-netmera-troubleshooting-and-support.txt": "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/troubleshooting-and-support",
    "netmera-user-guide-beginners-guide-to-netmera-faq.txt": "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/faqs",
    "netmera-user-guide-beginners-guide-to-netmera-app-dashboard.txt": "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/app-dashboard",
    "netmera-user-guide-messages-mobile-push-creating-a-mobile-push.txt": "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push",
    "netmera-user-guide-messages-mobile-push-creating-a-mobile-push-define-notification-content-what.txt": "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/define-notification-content-what",
    "netmera-user-guide-messages-mobile-push-creating-a-mobile-push-define-notification-content-what-advanced-ios-settings.txt": "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/define-notification-content-what/advanced-ios-settings",
    "netmera-user-guide-messages-email-sending-a-mail-campaign.txt": "https://user.netmera.com/netmera-user-guide/messages/email/sending-a-mail-campaign",
    "netmera-user-guide-messages-email-sending-a-mail-campaign-step-1-setup.txt": "https://user.netmera.com/netmera-user-guide/messages/email/sending-a-mail-campaign/step-1-setup",
    "netmera-user-guide-messages-sms-sms-onboarding.txt": "https://user.netmera.com/netmera-user-guide/messages/sms/sms-onboarding",
}

# 📁 Compound klasör fallback'leri
compound_sections = [
    "customer-data",
    "email-onboarding",
    "message-categories",
    "mobile-push",
    "push-notifications",
    "beginners-guide-to-netmera",
    "email",
    "sms",
]

# 📂 Üst seviye ve alt seviye klasörler — Sidebar yapısına göre
top_level_sections = {
    "messages": [
        "about-push-notifications",
        "mobile-push",
        "sms",
        "email",
        "automated-messages",
        "transactional-messages",
        "geofence-messages",
        "push-a-b-testing",
        "file-transfer-protocol-ftp-push",
        "multi-language-push",
        "recall-campaigns",
        "netmera-ai-text-generator",
    ],
    "customer-data": [
        "about-customer-data",
        "autotracking",
        "events",
        "profile-attributes",
    ],
    "beginners-guide-to-netmera": [
        "introduction-to-netmera",
        "integrating-netmera",
        "faq",
        "app-dashboard",
        "your-feedback",
        "design-guide",
        "troubleshooting-and-support",
    ],
}
def detect_language(text):
    try:
        lang_code = detect(text)
        if lang_code == "tr":
            return "Türkçe"
        else:
            return "English"
    except:
        return "English"  # Default

# 🔗 Dosya adını Netmera döküman linkine çevir
def filename_to_url(filename: str) -> str:
    # 🆕 0️⃣ Eğer bu bir FAQ embed'iyse (örnek: faq-email_quota_limits.json)
    if filename.startswith("faq-"):
        return "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/faqs"

    # 1️⃣ Manuel eşleşme varsa onu döndür
    if filename in file_to_url_map:
        return file_to_url_map[filename]

    # 2️⃣ Dosya adından "netmera-user-guide-" ve ".txt" kısımlarını ayıkla
    name = filename.replace(".txt", "")
    if name.startswith("netmera-user-guide-"):
        name = name[len("netmera-user-guide-") :]

    parts = name.split("-")
    base_url = "https://user.netmera.com/netmera-user-guide"

    # 3️⃣ Sidebar hiyerarşisine göre eşleştirme (messages, customer-data, etc.)
    for top_level, subfolders in top_level_sections.items():
        if parts[0] == top_level:
            rest = parts[1:]
            for i in range(2, 5):
                candidate = "-".join(rest[:i])
                if candidate in subfolders:
                    section = candidate
                    return f"{base_url}/{top_level}/{section}/{'-'.join(rest[i:])}"
            return f"{base_url}/{top_level}/{'-'.join(rest)}"

    # 4️⃣ Compound klasör fallback
    for i in range(2, 6):
        candidate = "-".join(parts[:i])
        if candidate in compound_sections:
            section = candidate
            rest = parts[i:]
            return f"{base_url}/{section}/{'-'.join(rest)}"

    # 5️⃣ Fallback
    return f"{base_url}/{'/'.join(parts)}"

    # 5️⃣ Hiçbiri eşleşmezse düz path döndür (fallback)
    return f"{base_url}/{'/'.join(parts)}"


# 🔎 Embed
def embed_question(question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Translate the question to English, only return the translation."},
            {"role": "user", "content": question},
        ],
    )
    translated = response.choices[0].message.content.strip()

    response_embed = client.embeddings.create(
        input=[translated], model="text-embedding-ada-002"
    )
    return np.array(response_embed.data[0].embedding, dtype=np.float32).reshape(1, -1), translated



# 📋 Log fonksiyonu
def log_interaction(question, answer, source_file, faiss_score):
    with open(
        "logs/conversation_log.csv", "a", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                datetime.now(),
                question,
                source_file,
                faiss_score,
                answer.replace("\n", " "),
            ]
        )


# 🤖 Yanıt üret
def ask_openai(question, context, lang="English"):
    prompt = f"""
You are NetmerianBot, a knowledgeable assistant specialized in Netmera's features and documentation.

You are given a document excerpt below. Your task is to answer the user's question using only the information in the document.

If the document provides relevant information:
- Give a clear, concise, assistant-style answer.
- Summarize key points naturally, as a helpful expert would.
- Do **not** copy the document verbatim — rephrase it.

If the answer is not covered in the content, simply say it's not available in the provided content. Do not mention "document" , not speculate or make assumptions.

DOCUMENT:
{context}

QUESTION:
{question}

ANSWER:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    english_answer = response.choices[0].message.content.strip()

    if lang == "Türkçe":
        translation = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Translate the following answer to Turkish:"},
                {"role": "user", "content": english_answer},
            ],
        )
        turkish_answer = translation.choices[0].message.content.strip()
        return turkish_answer
    else:
        return english_answer


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
        "If the ‘Send All’ option is selected for a push notification in the Netmera panel, will it be delivered to all users, even those who are not integrated with Netmera?",
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
    no_info_message = (
        "⚠️ There is not enough information on this topic. Please ask more clearly."
    )

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
    # ✅ Önce FAQ kontrolü
    faq_response = check_faq_match(user_input)
    
    if faq_response:
        answer = faq_response
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", answer))

    else:
        # 🌐 Dil algılama
        if not lang:
            lang = detect(user_input)
            if lang not in ["en", "tr"]:
                lang = "English"
            elif lang == "tr":
                lang = "Türkçe"
            else:
                lang = "English"
            st.info(f"🧠 Algılanan dil: {lang}")

        st.session_state.chat_history.append(("user", user_input))

        # 🔎 FAISS'ten en yakın 3 belgeyi getir
        embedding, translated_input = embed_question(user_input)
        distances, indices = index.search(embedding, k=5)
       # Tokenize user input
        tokenized_query = word_tokenize(user_input.lower())
        bm25_scores = bm25_model.get_scores(tokenized_query)
        
        candidate_docs = []
        for i, idx in enumerate(indices[0]):
            doc = texts[idx]
            faiss_score = distances[0][i]
            bm25_score = bm25_scores[idx]
            fuzzy_score = fuzz.partial_ratio(translated_input.lower(), doc["text"][:1000].lower())
            hybrid = compute_hybrid_score(doc, bm25_score, faiss_score, fuzzy_score)
            
            doc["faiss_score"] = faiss_score
            doc["bm25_score"] = bm25_score
            doc["fuzzy_score"] = fuzzy_score
            doc["hybrid_score"] = hybrid
            candidate_docs.append(doc)
        

        best_doc = max(candidate_docs, key=lambda d: d["hybrid_score"])

        top_docs = sorted(candidate_docs, key=lambda d: d["hybrid_score"], reverse=True)[:3]
        top_k_context = "\n\n---\n\n".join([doc["text"] for doc in top_docs])
        answer_text = ask_openai(user_input, top_k_context, lang)
        source_file = best_doc["source"]
        source_url = filename_to_url(source_file)

        if "Ekim 2023'e kadar olan veriler" in answer_text:
            answer = no_info_message
        else:
            answer = f"{answer_text}\n\n📄 **Kaynak belge**: [{source_file}]({source_url})"

        st.session_state.chat_history.append(("assistant", answer))
        log_interaction(user_input, answer, source_file, best_doc["faiss_score"])


# 💬 Geçmişi göster
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

st.markdown("---")
st.markdown("### 📊 Konuşma Kayıtları")

# Eğer log dosyası varsa indirilebilir olarak sun
if os.path.exists("logs/conversation_log.csv"):
    try:
        df_logs = pd.read_csv("logs/conversation_log.csv")
        st.download_button(
            label="📥 Logları CSV olarak indir",
            data=df_logs.to_csv(index=False).encode("utf-8"),
            file_name="conversation_log.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"Log dosyası okunurken hata oluştu: {e}")
else:
    st.info("Henüz herhangi bir log dosyası bulunmuyor.")