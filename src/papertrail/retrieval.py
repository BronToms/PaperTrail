import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from papertrail.corpus import Chunk

METHODS = ("bm25", "tfidf", "dense", "hybrid")


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in ENGLISH_STOP_WORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def to_dict(self):
        return {**asdict(self.chunk), "score": self.score}


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        method: str = "bm25",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks, self.method, self.model_name = chunks, method, model
        texts = [c.title + "\n" + c.text for c in chunks]
        self.counts = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avg_length = max(float(self.lengths.mean()), 1)
        df = Counter(t for c in self.counts for t in c)
        self.idf = {t: math.log(1 + (len(chunks) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
            self.matrix = self.vectorizer.fit_transform(texts)
        if method in {"dense", "hybrid"}:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model)
            self.embeddings = self.encoder.encode_document(texts, normalize_embeddings=True)

    def bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        for term in set(tokenize(query)):
            tf = np.array([c.get(term, 0) for c in self.counts])
            denominator = tf + 1.5 * (0.25 + 0.75 * self.lengths / self.avg_length)
            scores += self.idf.get(term, 0) * tf * 2.5 / denominator
        return scores

    def scores(self, query: str) -> np.ndarray:
        if self.method == "bm25":
            return self.bm25(query)
        if self.method == "tfidf":
            return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        vector = self.encoder.encode_query([query], normalize_embeddings=True)[0]
        dense = self.embeddings @ vector
        if self.method == "dense":
            return dense
        # RRF over at most 50 candidates from each retriever; scores are not probabilities.
        fused = np.zeros(len(self.chunks))
        for scores in (self.bm25(query), dense):
            ranked = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:50]
            for rank, i in enumerate(ranked, 1):
                fused[i] += 1 / (60 + rank)
        return fused

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        if not tokenize(query):
            return []
        scores = self.scores(query)
        return [
            Hit(self.chunks[i], float(scores[i]))
            for i in np.argsort(-scores, kind="stable")
            if scores[i] > 0
        ][:k]
