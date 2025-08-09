"""
NetmerianBot — A bilingual (Turkish/English) Streamlit-based chatbot for interacting with Netmera documentation using Retrieval-Augmented Generation (RAG).

Main Features:
- Language detection and translation
- Hybrid retrieval using FAISS, BM25, and Fuzzy Matching
- Answer generation via OpenAI Chat Completion API
- FAQ matching for fast predefined responses
- Source URL citation in each answer

The user query is embedded, matched with the most relevant documents, and passed to OpenAI to generate a natural-language answer.
"""

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
from langsmith import traceable
from langsmith.client import Client
from langsmith.wrappers import wrap_openai
import openai

client = wrap_openai(openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"]))


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

if "suggestion_buttons" not in st.session_state:
    st.session_state.suggestion_buttons = None

if "suggestions_cache" not in st.session_state:
    st.session_state.suggestions_cache = {}  

index = faiss.read_index(EMBEDDINGS_PATH)
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)
    
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_qa_map = json.load(f)

corpus = [doc["text"] for doc in texts]
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
bm25_model = BM25Okapi(tokenized_corpus)

def compute_hybrid_score(doc, norm_bm25, norm_faiss, norm_fuzzy):
    """
    Computes a weighted hybrid score by combining BM25, FAISS (vector similarity), and fuzzy string matching scores.

    Args:
        doc (dict): Document object.
        norm_bm25 (float): Normalized BM25 score.
        norm_faiss (float): Cosine similarity score (assumed normalized).
        norm_fuzzy (float): Fuzzy match score (0.0 to 1.0).

    Returns:
        float: Combined hybrid relevance score.
    """
    return BM25_WEIGHT * norm_bm25 + FUZZY_WEIGHT * norm_fuzzy + FAISS_WEIGHT * norm_faiss

def check_faq_match(translated_input, threshold=80):
    """
    Matches the user query with known FAQ questions using fuzzy string matching.

    Args:
        translated_input (str): English-translated user input.
        threshold (int): Similarity threshold for accepting a match.

    Returns:
        str or None: The matching FAQ answer and source link, or None if no match is found.
    """
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
    """
    Detects the language of the input text using langdetect.

    Args:
        text (str): Raw input text.

    Returns:
        str: "Türkçe" for Turkish, "English" otherwise.
    """
    try:
        lang_code = detect(text)
        return "Türkçe" if lang_code == "tr" else "English"
    except:
        return "English"  
    
def generate_questions_from_context(context, lang="English"):
    prompt = f"""
Based on the content below, generate 3 potential user questions someone might ask to retrieve this information. Write each question in a separate line.

CONTENT:
{context}
"""
    if lang == "Türkçe":
        prompt = f"""
Aşağıdaki içeriğe göre, bir kullanıcının bu bilgiyi sormak için sorabileceği 3 olası soruyu üret. Her soruyu ayrı satıra yaz.

İÇERİK:
{context}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().split("\n")

def filename_to_url(filename: str) -> str:
    """
    Converts a document filename into its corresponding Netmera documentation URL.

    Args:
        filename (str): Name of the .txt file.

    Returns:
        str: Public URL of the documentation page.
    """
    if filename.startswith("faq-"):
        return FAQ_URL
    if filename.startswith("netmera-user-guide-"):
        filename = filename[len("netmera-user-guide-"):]
    if filename.endswith(".txt"):
        filename = filename[:-4]
    url_path = filename.replace("-", "/")
    return f"{BASE_DOC_URL}/{url_path}"

def find_source_for_question(question_text):
    # İstersen burada önce çeviri yapabilirsin; çoğu embedding modelinde gerek olmayabilir.
    embedding = embed_question(question_text)[0]
    tokenized_q = word_tokenize(question_text.lower())

    bm25_scores_q = bm25_model.get_scores(tokenized_q)
    bm25_mean_q = np.mean(bm25_scores_q)
    bm25_std_q = np.std(bm25_scores_q) or 1.0

    best = None
    best_score = -1e9

    for idx, doc in enumerate(texts):
        doc_emb = index.reconstruct(idx)
        nd = np.linalg.norm(doc_emb)
        nq = np.linalg.norm(embedding)
        cos = np.dot(doc_emb, embedding) / (nd * nq) if nd and nq else 0.0

        bm25_norm = (bm25_scores_q[idx] - bm25_mean_q) / bm25_std_q
        fuzzy = fuzz.partial_ratio(question_text.lower(), doc["text"][:1000].lower()) / 100.0

        hybrid = compute_hybrid_score(doc, bm25_norm, cos, fuzzy)
        if hybrid > best_score:
            best_score = hybrid
            best = doc

    if not best:
        return None

    return best.get("url") or filename_to_url(best["source"])
def embed_question(translated_input):
    """
    Embeds the translated user input into a vector using OpenAI's embedding API.

    Args:
        translated_input (str): User query in English.

    Returns:
        np.ndarray: A (1, D) vector representing the input.
    """
    response_embed = client.embeddings.create(
        input=[translated_input], model=EMBEDDING_MODEL
    )
    return np.array(response_embed.data[0].embedding, dtype=np.float32).reshape(1, -1)


        
def suggest_similar_questions(user_input, faq_questions, top_n=3):
    scored = []
    for q in faq_questions:
        score = fuzz.partial_ratio(user_input.lower(), q.lower())
        scored.append((q, score))
    top_suggestions = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    return [q for q, _ in top_suggestions]

def generate_answerable_questions(context, user_lang="English", max_try=5):
    raw_questions = generate_questions_from_context(context, user_lang)
    validated = [] 

    for q in raw_questions[:max_try]:
        ans = ask_openai(q, context, user_lang)
        if ans: 
            src_url = find_source_for_question(q)  
            validated.append((q, ans))


    return validated[:3] 

@traceable(name="Ask OpenAI")
def ask_openai(question, context, lang="English"):
    """
    Uses OpenAI Chat Completion to generate an answer from the provided context.

    If the user’s language is Turkish, the English answer is translated back to Turkish.

    Returns:
        str or None: Final answer, or None if fallback/generic text is detected.
    """
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

    fallback_phrases = [
    "no relevant information",
    "not enough information",
    "unable to find",
    "i don't have information",
    "uygun bilgi mevcut değil",
    "bilgi bulunamadı",
    "yetersiz bilgi",
    "bu konuda bilgi yok",
    "cevap veremem"
     ]
    if not english_answer or any(p in english_answer.lower() for p in fallback_phrases):
        return None

    # 🇹🇷 Türkçeye çevir
    if lang == "Türkçe":
        translation = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": TURKISH_TRANSLATION_PROMPT},
                {"role": "user", "content": english_answer},
            ],
        )
        turkish_answer = translation.choices[0].message.content.strip()

        if not turkish_answer or any(p in turkish_answer.lower() for p in fallback_phrases):
            return None

        return turkish_answer

    return english_answer

lang_manual = st.toggle("Dil seç", value=False)
lang = st.radio("Dil / Language", ("Türkçe", "English"), horizontal=True) if lang_manual else None

st.set_page_config(page_title="NetmerianBot", layout="centered")
st.markdown("""
<style>
.answer-card{
  background:#111418; border:1px solid #1f2937; padding:16px; border-radius:12px; margin:8px 0;
}
.answer-header{ display:flex; gap:8px; align-items:center; margin-bottom:8px; }
.badge{
  font-size:12px; padding:2px 8px; border-radius:999px;
  background:#0b5; color:white; border:1px solid #0a4;
}
.question-pill{
  display:inline-block; padding:8px 12px; border-radius:10px;
  background:#0f172a; border:1px solid #1f2937; margin:4px 0;
}
.divider{ height:1px; background:#1f2937; margin:10px 0;}
.small{ color:#94a3b8; font-size:12px;}
.copy-btn{ float:right; font-size:12px; color:#94a3b8;}
</style>
""", unsafe_allow_html=True)
st.title("🤖 NetmerianBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestion_buttons" not in st.session_state:
    st.session_state.suggestion_buttons = None

faq_questions = FAQ_QUESTIONS_TR if lang == "Türkçe" else FAQ_QUESTIONS_EN
input_placeholder = "Bir soru yazın..." if lang == "Türkçe" else "Type a question..."
no_info_message = "⚠️ Bu konuda yeterli bilgi yok. Lütfen daha açık şekilde sorun." if lang == "Türkçe" else "⚠️ There is not enough information on this topic. Please ask more clearly."

cols = st.columns(2)
selected_question = None
for i, q in enumerate(faq_questions):
    if cols[i % 2].button(q, key=f"faq_{i}"):
        selected_question = q

user_input = st.chat_input(input_placeholder)
if "triggered_input" in st.session_state:
    user_input = st.session_state.triggered_input
    del st.session_state["triggered_input"]
if selected_question and not user_input:
    user_input = selected_question
if user_input and "chat_input" in st.session_state:
    last_user_message = st.session_state.chat_history[-1][1] if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user" else None
    if user_input == last_user_message:
        user_input = None  

if user_input and (len(st.session_state.chat_history) == 0 or user_input != st.session_state.chat_history[-1][1]):
 if len(user_input.strip().split()) <= 2:
        warning_msg = "⚠️ Lütfen daha açıklayıcı bir soru yazın. (En az 3 kelime giriniz.)" if lang == "Türkçe" else "⚠️ Please ask a more detailed question. (Minimum 3 words required.)"
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", warning_msg))
 else:
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
        with st.spinner("Yanıt hazırlanıyor…"):
         answer_text = ask_openai(user_input, top_k_context, lang)
        
        if answer_text is None:
           with st.status("Öneriler hazırlanıyor…", expanded=False) as status:
            validated_questions = generate_answerable_questions(top_k_context, lang)
            status.update(label="Öneriler hazır ✅", state="complete")
           if validated_questions:
              st.session_state.suggestions_cache.clear()
              st.session_state.suggestion_buttons = []

              source_file = best_doc["source"]
              source_url = best_doc.get("url") or filename_to_url(source_file)
  
              for q, a in validated_questions:
                st.session_state.suggestions_cache[q] = {
                "answer": a,
                "source_url": url or (best_doc.get("url") or filename_to_url(best_doc["source"]))
                 }
                st.session_state.suggestion_buttons.append(q)
           else:
               with st.spinner("Benzer sorular aranıyor…"):
       
                    fallback_suggestions = suggest_similar_questions(user_input, faq_questions, top_n=3)
                    validated_fallbacks = []
                    for q in fallback_suggestions:
                      faq_resp = check_faq_match(q)
                      if faq_resp:
                         st.session_state.suggestions_cache[q] = {
                             "answer": faq_resp,
                             "source_url": FAQ_URL,
                         }
                         validated_fallbacks.append(q)
                      else:
                           a = ask_openai(q, top_k_context, lang)
                           if a:
                                url_q = find_source_for_question(q) 
                                st.session_state.suggestions_cache[q] = {
                                "answer": a,
                                "source_url": url_q,  
                                 }
                                validated_fallbacks.append(q)


                    if validated_fallbacks:
                       st.session_state.suggestion_buttons = validated_fallbacks
                    else:
                       st.session_state.chat_history.append(("assistant", no_info_message))
                       st.session_state.suggestion_buttons = None
                
        else:
         source_file = best_doc["source"]
         source_url = best_doc.get("url") or filename_to_url(source_file)
         short_name = source_url.replace("https://user.netmera.com/netmera-user-guide/", "").replace("-", " ").replace("/", " > ").title()
         label_source = "📄 **Kaynak belge**" if lang == "Türkçe" else "📄 **Source document**"
         answer = f"{answer_text}\n\n{label_source}: [{short_name}]({source_url})"
         st.session_state.chat_history.append(("assistant", answer))

for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)
if st.session_state.suggestion_buttons:
    with st.chat_message("assistant"):
        title = "🤔 Şunu demek istemiş olabilir misiniz?" if lang == "Türkçe" else "🤔 Did you mean:"
        st.markdown(f"**{title}**")

    for i, suggestion in enumerate(st.session_state.suggestion_buttons):
        if st.button(suggestion, key=f"suggestion_{i}"):
            st.session_state.chat_history.append(("user", suggestion))

            cached = st.session_state.suggestions_cache.get(suggestion)
            if cached:
                label_source = "📄 **Kaynak belge**" if lang == "Türkçe" else "📄 **Source document**"
                short_name = cached["source_url"].replace("https://user.netmera.com/netmera-user-guide/", "").replace("-", " ").replace("/", " > ").title()
                answer = f"{cached['answer']}\n\n{label_source}: [{short_name}]({cached['source_url']})"
                st.session_state.chat_history.append(("assistant", answer))
            else:
                st.session_state.chat_history.append(("assistant", no_info_message))

            st.session_state.suggestion_buttons = None
            st.session_state.suggestions_cache.clear()
            st.rerun()