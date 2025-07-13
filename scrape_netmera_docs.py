import requests
from bs4 import BeautifulSoup
import os
import re

# Belge URL'leri
urls = [
    "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/introduction-to-netmera",
    "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/integrating-netmera",
    "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/app-dashboard",
    "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/design-guide",
    "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/troubleshooting-and-support",
    "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/faqs",
    "https://user.netmera.com/netmera-user-guide/terms-to-know",
    "https://user.netmera.com/netmera-user-guide/messages/about-push-notifications",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/define-message-type-setup",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/define-notification-content-what",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/define-notification-content-what/advanced-ios-settings",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/define-the-audience-who",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/define-campaign-schedule-when",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/creating-a-mobile-push/test-and-send-go",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/elements-of-push-notifications",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/message-categories",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/personalized-messages",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/button-sets",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/define-segment",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/add-custom-sounds",
    "https://user.netmera.com/netmera-user-guide/messages/mobile-push/add-image-or-video-to-push-notifications"
]

# Kayıt klasörünü oluştur
os.makedirs("data/documents", exist_ok=True)

for url in urls:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        # Dosya adını oluştur
        title = soup.title.string.strip().split("|")[0].strip()
        filename = f"data/documents/{title.replace(' ', '_').replace('/', '_').lower()}.txt"

        # Tüm metni al
        text = soup.get_text(separator="\n")

        # Gereksiz ifadeleri çıkar
        cleanup_phrases = [
            "Previous", "Next", "Last updated", "Was this helpful?",
            "Integrating Netmera", "Design Guide", "Introduction to Netmera"
        ]
        for phrase in cleanup_phrases:
            text = text.replace(phrase, "")

        # Boşlukları temizle
        text = re.sub(r"\n\s*\n", "\n", text).strip()

        # Dosyaya kaydet
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Cleaned & saved: {filename}")

    except Exception as e:
        print(f"❌ Error while processing {url}: {e}")