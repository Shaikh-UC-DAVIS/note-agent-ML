import psycopg2
from pgvector.psycopg2 import register_vector

class PostgresMetadataStorage:
    def __init__(self, conn_string):
        self.conn = psycopg2.connect(conn_string)
        register_vector(self.conn)

    def insert_chunk(self, chunk_id, text, token_count, embedding):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (id, text, token_count, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
                """,
                (chunk_id, text, token_count, embedding),
            )
        self.conn.commit()

    def search_vector(self, query_vec, limit=5):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text, embedding <-> %s AS distance
                FROM chunks
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (query_vec, query_vec, limit),
            )
            return cur.fetchall()

