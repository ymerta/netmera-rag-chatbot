EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"

BM25_WEIGHT = 0.4
FAISS_WEIGHT = 0.3
FUZZY_WEIGHT = 0.3


EMBEDDINGS_PATH = "data/embeddings/index.faiss"
TEXTS_PATH = "data/embeddings/texts.pkl"
FAQ_PATH = "data/faq_answers.json"


BASE_DOC_URL = "https://user.netmera.com/netmera-user-guide"
FAQ_URL = "https://user.netmera.com/netmera-user-guide/beginners-guide-to-netmera/faqs"


TRANSLATE_PROMPT = "Translate the question to English. If it's already in English, return it as-is. Only return the sentence."

SYSTEM_PROMPT = """
You are NetmerianBot, a knowledgeable assistant specialized in Netmera's features and documentation.

Your job is to answer the user's question using only the provided content. If the content contains relevant information, provide a clear, concise answer. 

Guidelines:
- Use only the content below.
- Do not mention training data or your knowledge cut-off.
- Rephrase and summarize naturally.
- If the content does not answer the question, respond with: "There is no relevant information available."
"""

TURKISH_TRANSLATION_PROMPT = (
    "You are a professional translator. Translate the following response to Turkish accurately and naturally. "
    "⚠️ However, do NOT translate technical terms like 'Send All', 'Push Notification', 'Segment', 'SDK', etc. "
    "Keep them exactly as they are. Do not add anything."
)

FAQ_QUESTIONS_TR = [
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

FAQ_QUESTIONS_EN = [
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