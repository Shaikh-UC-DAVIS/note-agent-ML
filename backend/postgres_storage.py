import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

class PostgresMetadataStorage:
    def __init__(self, conn_string):
        try:
            self.conn = psycopg2.connect(conn_string)
            register_vector(self.conn)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Postgres: {e}")

    def insert_chunk(self, chunk_id, text, token_count, embedding):
        if not isinstance(embedding, list):
            raise TypeError("Embedding must be a list")
        if not embedding or len(embedding) != 384:
            raise ValueError("Embedding must be a 384-dim list")
        
        # Convert list to a Postgres-compatible string: "[0.1, 0.2, ...]"
        vec_string = "[" + ",".join(map(str, embedding)) + "]"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (id, text, token_count, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (chunk_id, text, token_count, vec_string)
                )
            self.conn.commit()
        except psycopg2.Error as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to insert chunk {chunk_id}: {e}")


    def search_vector(self, query_vec, limit=5):
        # Convert numpy array to list
        if isinstance(query_vec, np.ndarray):
            query_vec = query_vec.tolist()

        if not query_vec:
            raise ValueError("Query vector cannot be empty")
        if len(query_vec) != 384:
            raise ValueError("Query vector must be 384-dim list")

        vec_string = "[" + ",".join(map(str, query_vec)) + "]"

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, text, embedding <-> %s::vector AS distance
                    FROM chunks
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (vec_string, vec_string, limit)
                )
                return cur.fetchall()
        except psycopg2.Error as e:
            raise RuntimeError(f"Vector search failed: {e}")
