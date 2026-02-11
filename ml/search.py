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
    def __init__(self, embedding_generator=None, graph=None):
        """
        Initialize the Hybrid Search Engine.
        
        Args:
            embedding_generator: Instance of EmbeddingGenerator (from ingestion).
            graph: Instance of KnowledgeGraph (from graph).
        """
        self.embedding_generator = embedding_generator
        self.graph = graph
        # Mock storage for chunks/vectors
        self.chunks: List[Dict] = [] 

    def index_chunk(self, chunk_id: str, text: str, vector: List[float]):
        """Adds a chunk to the search index (mock)."""
        self.chunks.append({
            "id": chunk_id,
            "text": text,
            "vector": vector
        })

    def _vector_search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """Performs a simple cosine similarity search (mock/inefficient for demo)."""
        if not self.chunks:
            return []
            
        results = []
        q_vec = np.array(query_vector)
        
        for chunk in self.chunks:
            c_vec = np.array(chunk['vector'])
            # Cosine similarity
            similarity = np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
            results.append(SearchResult(
                chunk_id=chunk['id'],
                text=chunk['text'],
                score=float(similarity),
                source='vector'
            ))
            
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _keyword_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Performs a simple keyword search."""
        results = []
        query_terms = set(query.lower().split())
        
        for chunk in self.chunks:
            text_lower = chunk['text'].lower()
            score = 0
            for term in query_terms:
                if term in text_lower:
                    score += 1
            
            if score > 0:
                results.append(SearchResult(
                    chunk_id=chunk['id'],
                    text=chunk['text'],
                    score=float(score * 10), # Boost keyword scores for visibility in this simple demo
                    source='keyword'
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search(self, query: str) -> List[SearchResult]:
        """
        Performs a hybrid search.
        """
        # 1. Generate query embedding
        if self.embedding_generator:
            # We mock the chunk/token structure for the query
            # In real usage, we'd have a simpler encode method
            # For now, let's assume embedding_generator.model works directly on strings if we access it
            query_vector = self.embedding_generator.model.encode(query)
            vector_results = self._vector_search(query_vector)
        else:
            vector_results = []
            
        # 2. Keyword search
        keyword_results = self._keyword_search(query)
        
        # 3. Merge and Rerank (Simple Interleaving/Deduplication)
        combined = {r.chunk_id: r for r in vector_results}
        for res in keyword_results:
            if res.chunk_id in combined:
                combined[res.chunk_id].score += res.score # Simple score fusion
                combined[res.chunk_id].source = "hybrid"
            else:
                combined[res.chunk_id] = res
                
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results
