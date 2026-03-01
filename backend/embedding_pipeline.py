import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:

    BATCH_SIZE = 50

    def __init__(self, conn_string, model_name="all-MiniLM-L6-v2"):
        self.conn = psycopg2.connect(conn_string)
        register_vector(self.conn)
        self.model = SentenceTransformer(model_name)

    def embed_spans_task(self, note_id):

        # 1–2. Load spans + filter embedding IS NULL
        spans = self._load_unembedded_spans(note_id)
        if not spans:
            print("No spans require embedding.")
            return

        # 3. Batch spans (50–100)
        batches = [
            spans[i:i + self.BATCH_SIZE]
            for i in range(0, len(spans), self.BATCH_SIZE)
        ]

        for batch in batches:
            ids = [s[0] for s in batch]
            texts = [s[1] for s in batch]

            # 4. Call embedding model
            embeddings = self._generate_embeddings(texts)

            # 5. Store vectors
            self._store_embeddings(ids, embeddings)

        # 6. Ensure pgvector HNSW index exists
        self._ensure_hnsw_index()

        # 7. Update note status
        self._mark_note_embedded(note_id)

        # 8. Trigger next stage
        self._trigger_structure_extraction(note_id)

        print("Embedding complete.")


    def _load_unembedded_spans(self, note_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text
                FROM spans
                WHERE note_id = %s
                AND embedding IS NULL;
                """,
                (str(note_id),)
            )
            return cur.fetchall()


    def _generate_embeddings(self, texts):
        return self.model.encode(texts).tolist()

    def _store_embeddings(self, span_ids, embeddings):
        try:
            with self.conn.cursor() as cur:

                records = [
                    (
                        "[" + ",".join(map(str, emb)) + "]",
                        span_id
                    )
                    for span_id, emb in zip(span_ids, embeddings)
                ]

                cur.executemany(
                    """
                    UPDATE spans
                    SET embedding = %s::vector
                    WHERE id = %s;
                    """,
                    records
                )

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Embedding write failed: {e}")


    def _ensure_hnsw_index(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS spans_embedding_hnsw
                ON spans
                USING hnsw (embedding vector_l2_ops);
                """
            )
        self.conn.commit()

    def _mark_note_embedded(self, note_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE notes
                    SET status = 'embedded'
                    WHERE id = %s;
                    """,
                    (note_id,)
                )
            self.conn.commit()
        except Exception:
            # ML repo may run without orchestration DB
            self.conn.rollback()


    def _trigger_structure_extraction(self, note_id):
        # Celery hook placeholder
        print(f"Trigger extract_structure({note_id})")