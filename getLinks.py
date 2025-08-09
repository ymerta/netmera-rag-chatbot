"""
Netmera Documentation Scraper

This script scrapes all user-facing documentation pages from the Netmera User Guide sidebar and saves the cleaned
text content of each page into individual .txt files. Each file includes the source URL for traceability.

Steps:
1. Fetch all sidebar links under /netmera-user-guide/
2. Extract only the main readable content of each page
3. Clean the text by removing navigation, update info, and UI noise
4. Save as plain text files with structured filenames

Output:
- Text files saved under `data/documents/`, prefixed with `netmera-user-guide-`
- Each file begins with `[SOURCE_URL]: <url>` as the first line
"""
import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://user.netmera.com"
START_PAGE = f"{BASE_URL}/netmera-user-guide/"
SAVE_FOLDER = "data/documents"

os.makedirs(SAVE_FOLDER, exist_ok=True)

def get_all_sidebar_links():
    """
    Parses the Netmera User Guide sidebar and returns all internal documentation URLs.

    Returns:
        List[str]: A sorted list of full URLs to documentation pages.
    """
    response = requests.get(START_PAGE)
    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    for a in soup.select("aside a[href]"):  
        href = a["href"]
        if href.startswith("/netmera-user-guide") and "http" not in href:
            full_url = f"{BASE_URL}{href}"
            links.add(full_url)

    return sorted(links)

def get_main_content(html):
    """
    Extracts the main content block from a documentation HTML page.

    Args:
        html (str): Raw HTML content of the page.

    Returns:
        str: Extracted text content from <main> or <article> tag, fallback to full page text if not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("article")
    if main:
        return main.get_text(separator="\n").strip()
    return soup.get_text(separator="\n").strip()

REMOVE_PHRASES = [
    "Netmera User Guide", "Ctrl", "K", "Netmera Docs", "More", "⚡",
    "Was this helpful?", "Copy", "Previous", "Next", "Last updated",
    "On this page"
]

def clean_text(text):
    """
    Cleans the extracted page text by removing unwanted boilerplate phrases and short lines.

    Args:
        text (str): Raw text extracted from HTML.

    Returns:
        str: Cleaned, line-separated text suitable for embedding and retrieval.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(p in line for p in REMOVE_PHRASES):
            continue
        if len(line) <= 2:
            continue  
        cleaned.append(line)

    return "\n".join(cleaned)

def url_to_filename(url: str) -> str:
    """
    Converts a documentation URL to a filesystem-safe filename.

    Args:
        url (str): Full Netmera URL starting with BASE_URL.

    Returns:
        str: Filename string prefixed with 'netmera-user-guide-' and using '-' as separators.
    """
    path = url.replace(BASE_URL + "/netmera-user-guide/", "")
    return "netmera-user-guide-" + path.replace("/", "-") + ".txt"


all_links = get_all_sidebar_links()
print(f"✅ {len(all_links)} belge bulundu. İndiriliyor...")

for url in all_links:
    try:
        r = requests.get(url)
        r.raise_for_status()
        raw_text = get_main_content(r.text)
        cleaned_text = clean_text(raw_text)
        filename = url_to_filename(url)

        with open(os.path.join(SAVE_FOLDER, filename), "w", encoding="utf-8") as f:
            f.write(f"[SOURCE_URL]: {url}\n")
            f.write(cleaned_text)

        print(f"Kaydedildi: {filename}")
    except Exception as e:
        print(f"Hata ({url}): {e}")