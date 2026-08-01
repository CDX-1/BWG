import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parallax.config import KNOWLEDGE_DIR


def load_knowledge_chunks() -> list[dict]:
    chunks = []
    for fname in os.listdir(KNOWLEDGE_DIR):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(KNOWLEDGE_DIR, fname)
        with open(path) as f:
            text = f.read()
        # Split on section headers like [TAG §x.y]
        sections = re.split(r'\n(?=\[)', text.strip())
        for section in sections:
            section = section.strip()
            if section:
                chunks.append({
                    "source": fname.replace(".md", ""),
                    "text": section,
                })
    return chunks


def retrieve(query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    if not chunks:
        return []
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in indices]
