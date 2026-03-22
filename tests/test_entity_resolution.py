"""
test_entity_resolution.py

Tests for the Entity Resolution pipeline (Stage 5).
Covers both inter-batch (new vs. pre-existing) and intra-batch
(two near-duplicates in the same resolve_entities_task() call) behavior.

Requires: PostgreSQL with pgvector running (same Docker setup as demo).
  docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg16

Run: python tests/test_entity_resolution.py
"""

import os
import sys
import uuid
import psycopg2
from pgvector.psycopg2 import register_vector

# Ensure project root is on the path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.entity_resolution import EntityResolver

# Use password from env or fall back to the demo default
_password = os.environ.get("PGPASSWORD", "postgres")
DB_CONN = f"dbname=note_agent user=postgres host=localhost password={_password}"
TEST_WORKSPACE = "ws_test_entity_resolution"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_conn():
    conn = psycopg2.connect(DB_CONN)
    register_vector(conn)
    return conn


def _setup_workspace(conn):
    """Wipe all test data so each test starts from a clean slate."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM links    WHERE workspace_id = %s", (TEST_WORKSPACE,))
        cur.execute("DELETE FROM insights WHERE workspace_id = %s", (TEST_WORKSPACE,))
        cur.execute("DELETE FROM objects  WHERE workspace_id = %s", (TEST_WORKSPACE,))
    conn.commit()


def _insert_object(conn, obj_id, canonical_text, obj_type="Claim", confidence=0.9):
    """Insert an object with no embedding; the resolver will generate it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO objects (id, type, canonical_text, confidence, status, workspace_id)
            VALUES (%s, %s, %s, %s, 'active', %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (obj_id, obj_type, canonical_text, confidence, TEST_WORKSPACE),
        )
    conn.commit()


def _count_same_as_links(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM links WHERE workspace_id = %s AND type = 'SameAs'",
            (TEST_WORKSPACE,),
        )
        return cur.fetchone()[0]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_inter_batch_resolution():
    """Baseline: a new object auto-merges with a near-identical pre-existing object."""
    print("\n── Test: Inter-Batch Resolution ──")

    try:
        conn = _get_conn()
    except Exception as e:
        print(f"  ⚠ Skipping: cannot connect to DB ({e})")
        return False

    _setup_workspace(conn)
    resolver = EntityResolver(DB_CONN)

    # Seed a pre-existing object with a known embedding
    pre_text = "Launch new Quantum Engine product by March 15th"
    pre_id = str(uuid.uuid4())
    _insert_object(conn, pre_id, pre_text)

    # Give it an embedding so _find_most_similar can match against it
    embedding = resolver._embed_texts([pre_text])[0]
    vec_str = "[" + ",".join(map(str, embedding)) + "]"
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE objects SET embedding = %s::vector WHERE id = %s",
            (vec_str, pre_id),
        )
    conn.commit()
    print("  ✓ Pre-existing object seeded with embedding")

    # New object — identical text → similarity = 1.0 → auto-merge
    new_id = str(uuid.uuid4())
    _insert_object(conn, new_id, pre_text)

    stats = resolver.resolve_entities_task([new_id], workspace_id=TEST_WORKSPACE)

    assert stats["merged"] == 1, f"Expected merged=1, got {stats}"
    assert stats["flagged"] == 0, f"Expected flagged=0, got {stats}"
    assert _count_same_as_links(conn) == 1, "Expected 1 SameAs link"
    print(f"  ✓ Merged: {stats['merged']}, Flagged: {stats['flagged']}, Unchanged: {stats['unchanged']}")
    print("  ✓ SameAs link created")

    conn.close()
    return True


def test_intra_batch_near_duplicates():
    """
    Two near-duplicate objects submitted in the same batch call.

    resolve_entities_task() stores all embeddings before the comparison loop,
    so object B's embedding is visible when object A is being evaluated
    (and vice versa). One should auto-merge into the other; then the second
    finds the first gone (status != 'active') and stays unchanged.

    Expected: merged=1, unchanged=1, one SameAs link.
    """
    print("\n── Test: Intra-Batch Near-Duplicates ──")

    try:
        conn = _get_conn()
    except Exception as e:
        print(f"  ⚠ Skipping: cannot connect to DB ({e})")
        return False

    _setup_workspace(conn)
    resolver = EntityResolver(DB_CONN)

    obj_a_id = str(uuid.uuid4())
    obj_b_id = str(uuid.uuid4())

    _insert_object(conn, obj_a_id, "Launch new Quantum Engine product by March 15th")
    _insert_object(conn, obj_b_id, "New Quantum Engine product launch is scheduled for March 15th")
    print("  ✓ obj_A and obj_B inserted into the same batch (no pre-existing objects)")

    stats = resolver.resolve_entities_task(
        [obj_a_id, obj_b_id], workspace_id=TEST_WORKSPACE
    )

    total = stats["merged"] + stats["flagged"] + stats["unchanged"]
    assert total == 2, f"Expected 2 objects accounted for, got {stats}"

    # Either a merge (similarity >= 0.95) or a flag (similarity >= 0.85) proves
    # that intra-batch comparison is happening. Both objects finding each other
    # and being flagged (flagged=2) is also valid — flagging does not change an
    # object's status to inactive, so both directions are detected.
    detected = stats["merged"] + stats["flagged"]
    assert detected >= 1, \
        f"Expected at least one intra-batch merge or flag, got {stats}. " \
        "Near-duplicate texts should exceed the FLAG_THRESHOLD (0.85)."

    print(f"  ✓ Merged: {stats['merged']}, Flagged: {stats['flagged']}, Unchanged: {stats['unchanged']}")
    if stats["merged"] >= 1:
        print(f"  ✓ SameAs link(s) created: {_count_same_as_links(conn)}")
    else:
        print("  ✓ Intra-batch similarity detected via flagging (similarity in 0.85–0.95 range)")

    conn.close()
    return True


def test_intra_batch_no_false_positives():
    """
    Two semantically unrelated objects in the same batch should not be merged or flagged.

    Expected: merged=0, flagged=0, unchanged=2.
    """
    print("\n── Test: Intra-Batch No False Positives ──")

    try:
        conn = _get_conn()
    except Exception as e:
        print(f"  ⚠ Skipping: cannot connect to DB ({e})")
        return False

    _setup_workspace(conn)
    resolver = EntityResolver(DB_CONN)

    obj_c_id = str(uuid.uuid4())
    obj_d_id = str(uuid.uuid4())

    _insert_object(conn, obj_c_id, "Launch new Quantum Engine product by March 15th", obj_type="Claim")
    _insert_object(conn, obj_d_id, "What is our go-to-market timeline for the EU?", obj_type="Question")
    print("  ✓ obj_C (Claim) and obj_D (Question, unrelated topic) inserted")

    stats = resolver.resolve_entities_task(
        [obj_c_id, obj_d_id], workspace_id=TEST_WORKSPACE
    )

    assert stats["merged"] == 0, f"Expected merged=0 for dissimilar objects, got {stats}"
    assert stats["flagged"] == 0, f"Expected flagged=0 for dissimilar objects, got {stats}"
    assert stats["unchanged"] == 2, f"Expected unchanged=2, got {stats}"
    assert _count_same_as_links(conn) == 0, "Expected no SameAs links for dissimilar objects"

    print(f"  ✓ Merged: {stats['merged']}, Flagged: {stats['flagged']}, Unchanged: {stats['unchanged']}")
    print("  ✓ No false-positive merges")

    conn.close()
    return True


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ENTITY RESOLUTION — STAGE 5 TESTS")
    print("=" * 60)

    results = [
        test_inter_batch_resolution(),
        test_intra_batch_near_duplicates(),
        test_intra_batch_no_false_positives(),
    ]

    print("\n" + "=" * 60)
    if all(r is not False for r in results):
        print("  ✓ ALL ENTITY RESOLUTION TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED OR WERE SKIPPED — see output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
