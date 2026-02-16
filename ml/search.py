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
    """
    Hybrid Search Engine for Note Agent
    Combines vector search (semantic) with optional keyword search
    """

    def __init__(self, embedding_generator, graph, storage=None):
        """
        embedding_generator: object with method generate_embeddings(chunks) or encode(text)
        graph: knowledge graph object
        storage: optional PostgresMetadataStorage instance
        """
        self.embedding_generator = embedding_generator
        self.graph = graph
        self.storage = storage
        self.chunks = []  # fallback in-memory storage if storage is None

    def index_chunk(self, chunk_id, text, vector, token_count=0):
        """
        Store chunk text and vector either in DB or in-memory list
        """
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
        """
        Perform vector similarity search using storage if available
        Returns: list of dicts with id, text, score
        """
        if self.storage:
            # Use Postgres + pgvector
            rows = self.storage.search_vector(query_vec, limit=top_k)
            results = []
            for row in rows:
                # row = (id, text, distance)
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
        """
        Simple keyword search fallback (in-memory)
        Returns list of dicts with id, text, score
        """
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
        """
        Public search method combining vector and keyword search
        Returns: list of dicts with id, text, score
        """
        # 1. Vector search
        query_vec = self.embedding_generator.model.encode(query_str)
        vector_results = self._vector_search(query_vec, top_k=top_k)

        # 2. Keyword search (in-memory fallback)
        keyword_results = self._keyword_search(query_str, top_k=top_k)

        # 3. Merge results (basic fusion)
        merged = {}
        for r in vector_results + keyword_results:
            if r["id"] not in merged:
                merged[r["id"]] = r
            else:
                # if found in both, boost score
                merged[r["id"]]["score"] += r["score"] * 0.5

        # 4. Sort merged results
        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        return sorted_results[:top_k]
