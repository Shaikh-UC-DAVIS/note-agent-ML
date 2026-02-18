from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    source: str # 'vector', 'keyword', 'graph'

class HybridSearchEngine:

    def __init__(self, embedding_generator, graph, storage=None):

        self.embedding_generator = embedding_generator
        self.graph = graph
        self.storage = storage
        self.chunks = []

    def index_chunk(self, chunk_id, text, vector, token_count=0):

        if self.storage:
            # Store in PostgreSQL
            self.storage.insert_chunk(
                chunk_id,
                text,
                token_count,
                vector
            )
        else:
            # fallback to in-memory
            self.chunks.append({
                "id": chunk_id,
                "text": text,
                "vector": vector,
                "token_count": token_count
            })

    def _vector_search(self, query_vec, top_k=5):

        if self.storage:
            # Use Postgres + pgvector
            rows = self.storage.search_vector(query_vec, limit=top_k)
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "score": 1 - row[2]  # convert distance to similarity score
                })
            return results
        else:
            # fallback in-memory search
            results = []
            for chunk in self.chunks:
                dist = np.linalg.norm(np.array(chunk["vector"]) - np.array(query_vec))
                results.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": 1 - dist
                })
            # sort by descending score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def _keyword_search(self, query_str, top_k=5):

        results = []
        for chunk in self.chunks:
            score = sum(1 for word in query_str.split() if word.lower() in chunk["text"].lower())
            if score > 0:
                results.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": float(score)
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search(self, query_str, top_k=5):

        # 1. Vector search
        query_vec = self.embedding_generator.model.encode(query_str)
        vector_results = self._vector_search(query_vec, top_k=top_k)

        # 2. Keyword search
        keyword_results = self._keyword_search(query_str, top_k=top_k)

        # 3. Merge and Rerank
        merged = {}
        for r in vector_results + keyword_results:
            if r["id"] not in merged:
                merged[r["id"]] = r
            else:
                merged[r["id"]]["score"] += r["score"] * 0.5

        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        return sorted_results[:top_k]
