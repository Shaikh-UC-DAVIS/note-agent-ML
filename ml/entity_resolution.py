import uuid
import json
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class EntityResolver:

    BATCH_SIZE = 50
    AUTO_MERGE_THRESHOLD = 0.95   # similarity >= this → merged_into_{id}
    FLAG_THRESHOLD = 0.85         # similarity >= this → consolidation_opportunity insight

    def __init__(self, conn_string, model_name="all-MiniLM-L6-v2"):
        self.conn = psycopg2.connect(conn_string)
        register_vector(self.conn)
        self.model = SentenceTransformer(model_name)

    def resolve_entities_task(self, new_obj_ids: list, workspace_id: str) -> dict:
        """
        Entry point. Compares each newly inserted object (identified by new_obj_ids)
        against all other active objects in the workspace and applies merge/flag logic.
        Returns summary stats: {merged, flagged, unchanged}.
        """
        if not new_obj_ids:
            return {"merged": 0, "flagged": 0, "unchanged": 0}

        # 1. Load newly inserted objects from DB
        objects = self._load_objects_by_ids(new_obj_ids)
        if not objects:
            return {"merged": 0, "flagged": 0, "unchanged": 0}

        # 2. Generate embeddings for canonical_text of each new object
        texts = [obj["canonical_text"] for obj in objects]
        embeddings = self._embed_texts(texts)

        # 3. Persist embeddings back to objects table
        self._store_object_embeddings([obj["id"] for obj in objects], embeddings)

        # 4. Compare each new object against pre-existing objects in the workspace
        merged = 0
        flagged = 0
        unchanged = 0

        for obj, embedding in zip(objects, embeddings):
            match = self._find_most_similar(embedding, [obj["id"]], workspace_id)
            if match is None:
                unchanged += 1
                continue

            match_id, similarity = match
            if similarity >= self.AUTO_MERGE_THRESHOLD:
                self._auto_merge(obj["id"], match_id, workspace_id)
                merged += 1
            elif similarity >= self.FLAG_THRESHOLD:
                self._flag_for_review(obj["id"], match_id, similarity, workspace_id)
                flagged += 1
            else:
                unchanged += 1

        return {"merged": merged, "flagged": flagged, "unchanged": unchanged}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_objects_by_ids(self, obj_ids: list) -> list:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical_text, type
                FROM objects
                WHERE id = ANY(%s) AND status = 'active'
                """,
                (obj_ids,)
            )
            rows = cur.fetchall()
        return [{"id": r[0], "canonical_text": r[1], "type": r[2]} for r in rows]

    def _embed_texts(self, texts: list) -> list:
        return self.model.encode(texts).tolist()

    def _store_object_embeddings(self, obj_ids: list, embeddings: list):
        try:
            with self.conn.cursor() as cur:
                records = [
                    ("[" + ",".join(map(str, emb)) + "]", obj_id)
                    for obj_id, emb in zip(obj_ids, embeddings)
                ]
                cur.executemany(
                    """
                    UPDATE objects
                    SET embedding = %s::vector
                    WHERE id = %s
                    """,
                    records
                )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Embedding write failed: {e}")

    def _find_most_similar(self, embedding, exclude_ids: list, workspace_id: str):
        """
        Returns (match_id, similarity) for the most similar pre-existing object
        above FLAG_THRESHOLD, or None if no such object exists.
        """
        vec_str = "[" + ",".join(map(str, embedding)) + "]"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                FROM objects
                WHERE workspace_id = %s
                  AND status = 'active'
                  AND id != ALL(%s)
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (vec_str, workspace_id, exclude_ids, vec_str)
            )
            row = cur.fetchone()

        if row is None:
            return None
        match_id, similarity = row
        if similarity < self.FLAG_THRESHOLD:
            return None
        return (match_id, similarity)

    def _auto_merge(self, src_id: str, dst_id: str, workspace_id: str):
        """Mark src as merged into dst and create a SameAs link."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE objects SET status = %s WHERE id = %s",
                    (f"merged_into_{dst_id}", src_id)
                )
                cur.execute(
                    """
                    INSERT INTO links (id, workspace_id, src_object_id, dst_object_id, type, confidence)
                    VALUES (%s, %s, %s, %s, 'SameAs', 1.0)
                    """,
                    (str(uuid.uuid4()), workspace_id, src_id, dst_id)
                )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Auto-merge failed: {e}")

    def _flag_for_review(self, src_id: str, dst_id: str, similarity: float, workspace_id: str):
        """Write a consolidation_opportunity insight for near-duplicate objects."""
        severity = "high" if similarity > 0.9 else "medium"
        payload = json.dumps({
            "src_id": src_id,
            "dst_id": dst_id,
            "similarity": round(similarity, 4),
            "reason": "Vector similarity above consolidation threshold"
        })
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO insights (id, workspace_id, type, severity, status, payload)
                    VALUES (%s, %s, 'consolidation_opportunity', %s, 'new', %s)
                    """,
                    (str(uuid.uuid4()), workspace_id, severity, payload)
                )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Flag for review failed: {e}")
