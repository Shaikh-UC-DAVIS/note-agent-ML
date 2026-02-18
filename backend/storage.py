from typing import List, Tuple
import psycopg
import numpy as np


class PostgresMetadataStorage:
    def __init__(self, conn_string: str, embedding_dim: int = 384):
        self.conn = psycopg.connect(conn_string)
        self.embedding_dim = embedding_dim

    def _to_list(self, embedding):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        if not isinstance(embedding, list):
            raise TypeError("Embedding must be list or numpy array")

        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding must be dimension {self.embedding_dim}")

        return embedding

    def insert_chunk(self, chunk_id, text, token_count, embedding):
        embedding = self._to_list(embedding)

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (id, text, token_count, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (chunk_id, text, token_count, embedding)
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def search_vector(self, query_vec, limit=5) -> List[Tuple]:
        query_vec = self._to_list(query_vec)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text, embedding <-> %s AS distance
                FROM chunks
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (query_vec, query_vec, limit)
            )
            return cur.fetchall()
