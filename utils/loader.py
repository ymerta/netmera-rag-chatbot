import os

def load_txt_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines and lines[0].startswith("[SOURCE_URL]: "):
                    url = lines[0].replace("[SOURCE_URL]: ", "").strip()
                    text = "".join(lines[1:]).strip()
                else:
                    url = None
                    text = "".join(lines).strip()
                documents.append({
                    "text": text,
                    "source": filename,
                    "url": url
                })
    return documents