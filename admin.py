import streamlit as st
import os
import shutil
import subprocess

st.set_page_config(page_title="🛠️ Yönetim Paneli", layout="centered")
st.title("📂 NetmerianBot - Admin Panel")

uploaded_file = st.file_uploader("Yeni .txt belgesi yükleyin", type=["txt"])

if uploaded_file:
    os.makedirs("data/new_docs", exist_ok=True)
    file_path = os.path.join("data/new_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Dosya yüklendi: {uploaded_file.name}")

    if st.button("Embed ve Güncelle"):
        with st.spinner("🔄 Belgeler embed ediliyor..."):
            result = subprocess.run(["python", "update_index.py"], capture_output=True, text=True)
            st.text(result.stdout)
        st.success("✅ Güncelleme tamamlandı.")