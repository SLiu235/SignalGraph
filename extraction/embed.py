"""
extraction.embed — embeddings for filings/news/transcripts. Default backend is
tfidf (offline, deterministic, no API); a sentence-transformers / openai backend
can be swapped in config. TF-IDF keeps retrieval reproducible for the synthetic set.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfIndex:
    """Deterministic TF-IDF index over a corpus of passages."""

    def __init__(self, passages: list[dict]):
        # passages: [{"id":..., "text":..., "meta":{...}}]
        self.ids = [p["id"] for p in passages]
        self.docs = [tokenize(p["text"]) for p in passages]
        self.n = len(self.docs)
        self.df = Counter()
        for toks in self.docs:
            for t in set(toks):
                self.df[t] += 1
        self._vectorize()

    def _vectorize(self):
        self.vecs = np.zeros((self.n, len(self.df)))
        vocab = {t: i for i, t in enumerate(self.df)}
        for di, toks in enumerate(self.docs):
            tf = Counter(toks)
            for t, c in tf.items():
                if t in vocab:
                    idf = np.log((1 + self.n) / (1 + self.df[t])) + 1.0
                    self.vecs[di, vocab[t]] = c * idf
        norms = np.linalg.norm(self.vecs, axis=1)
        norms[norms == 0] = 1.0
        self.vecs /= norms[:, None]

    def query(self, text: str, top_k: int = 8) -> list[dict]:
        toks = tokenize(text)
        vocab = {t: i for i, t in enumerate(self.df)}
        q = np.zeros(len(self.df))
        for t, c in Counter(toks).items():
            if t in vocab:
                q[vocab[t]] = c
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        scores = self.vecs @ q
        order = np.argsort(-scores)[:top_k]
        return [{"id": self.ids[i], "score": round(float(scores[i]), 4)} for i in order]


def build_index(passages: list[dict]) -> TfidfIndex:
    return TfidfIndex(passages)
