"""Três recuperadores sobre os chunks do livro:

  - BM25:   busca lexical (Okapi BM25), implementada aqui para controlar a tokenização
            de identificadores do FreeRTOS (xQueueSendToBack → x, queue, send, to, back).
  - Dense:  embeddings multilíngues (intfloat/multilingual-e5-small, roda em CPU).
            Permite perguntar em português sobre um livro em inglês.
  - Hybrid: combina os dois rankings com Reciprocal Rank Fusion (RRF).

Com ~400 chunks, a busca vetorial é um produto de matrizes em NumPy (< 1 ms);
um banco vetorial (FAISS, Chroma, pgvector) só se paga com dezenas de milhares de chunks.
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

STOPWORDS = set(
    "a an the of to in on for and or is are be been was were it its this that these those with as by at "
    "from can will does do how what which when why who if not no than then there their they you your into "
    "i e o os as um uma de da do das dos em no na nos nas para por com que se ao à é ser ou".split()
)
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"


def load_chunks(path="data/chunks.jsonl"):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def tokenize(text: str):
    tokens = []
    for word in re.findall(r"[A-Za-z0-9_À-ÿ]+", text):
        low = word.lower()
        if low not in STOPWORDS:
            tokens.append(low)
        # identificadores camelCase / snake_case: adiciona as partes também
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word.replace("_", " "))
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts if p.lower() not in STOPWORDS and len(p) > 1)
    return tokens


class BM25:
    name = "bm25"

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lens.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            for t in q:
                tf = d.get(t)
                if tf:
                    s[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return s

    def search(self, query, k=5):
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in idx]


class Dense:
    name = "dense"

    def __init__(self, chunks, model_name=DEFAULT_DENSE_MODEL, cache_dir="data", encoder=None):
        self.chunks = chunks
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name)
        self.encoder = encoder
        self.e5 = "e5" in model_name  # modelos e5 exigem os prefixos "query: " / "passage: "
        texts = [("passage: " if self.e5 else "") + c["text"] for c in chunks]
        key = hashlib.md5((model_name + "".join(c["id"] + c["text"] for c in chunks)).encode()).hexdigest()[:10]
        cache = Path(cache_dir) / f"emb_{model_name.split('/')[-1]}_{key}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.encode(texts, batch_size=32, show_progress_bar=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def encode(self, texts, **kw):
        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True, **kw), dtype=np.float32)

    def scores(self, query):
        q = self.encode([("query: " if self.e5 else "") + query])[0]
        return self.emb @ q

    def search(self, query, k=5):
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in idx]


class Hybrid:
    """Reciprocal Rank Fusion: score = Σ 1 / (k_rrf + posição). Não exige calibrar escalas."""

    name = "hybrid"

    def __init__(self, retrievers, k_rrf=60, depth=50):
        self.retrievers, self.k_rrf, self.depth = retrievers, k_rrf, depth
        self.chunks = retrievers[0].chunks

    def search(self, query, k=5):
        fused = np.zeros(len(self.chunks))
        for r in self.retrievers:
            order = np.argsort(-r.scores(query))[: self.depth]
            for rank, i in enumerate(order):
                fused[i] += 1.0 / (self.k_rrf + rank + 1)
        idx = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(fused[i])) for i in idx]


def build_retriever(kind="hybrid", chunks=None, model_name=DEFAULT_DENSE_MODEL):
    chunks = chunks or load_chunks()
    if kind == "bm25":
        return BM25(chunks)
    if kind == "dense":
        return Dense(chunks, model_name)
    return Hybrid([BM25(chunks), Dense(chunks, model_name)])
