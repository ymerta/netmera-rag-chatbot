import os

def load_txt_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append((filename, content))  # 👈 dosya adıyla birlikte dön
    return documents