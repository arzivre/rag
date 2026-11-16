import os

os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from cli.lib.search_utils import load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map: dict = {}
        self.embeddings_path = Path("src/cache/movie_embeddings.npy")

    def _set_documents(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

    def build_embedding(self, documents):
        self._set_documents(documents)
        strings = [f"{d['title']}: {d['description']}" for d in documents]
        self.embeddings = self.model.encode(strings, show_progress_bar=True)
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self._set_documents(documents)
        if self.embeddings_path.exists():
            cached = np.load(self.embeddings_path)
            if len(cached) == len(documents):
                self.embeddings = cached
                return self.embeddings
        return self.build_embedding(documents)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        return self.model.encode([text])[0]

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        qry_emb = self.generate_embedding(query)
        similarities = []
        for doc_embed, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(qry_emb, doc_embed)
            similarities.append((_similarity, doc))
        similarities.sort(key=lambda x: x[0], reverse=True)

        res = []
        for sc, doc in similarities[:limit]:
            res.append(
                {"score": sc, "title": doc["title"], "description": doc["description"]}
            )
        return res


def search(query, limit):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    search_results = ss.search(query, limit)
    for idx, res in enumerate(search_results):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        print(res["description"][:100])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
