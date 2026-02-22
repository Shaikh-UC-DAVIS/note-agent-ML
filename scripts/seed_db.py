
import os
import sys
import psycopg2
import json
import uuid
import random
from typing import List

# Add parent directory to path to import backend modules if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DB_CONFIG = "dbname=note_agent user=postgres password=postgres host=localhost"

def get_conn():
    try:
        return psycopg2.connect(DB_CONFIG)
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        sys.exit(1)

def run_seed():
    conn = get_conn()
    cur = conn.cursor()

    print("🌱 Seeding Database...")

    # 0. Apply Schema (Since we don't have psql CLI)
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'schema.sql')
    try:
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        cur.execute(schema_sql)
        print("✅ Schema applied successfully.")
    except Exception as e:
        print(f"⚠️ Failed to apply schema: {e}")
        # Continue anyway, tables might exist

    # Clear existing data (Optional, for development)
    cur.execute("TRUNCATE TABLE insights, links, object_mentions, objects, spans, files, notes CASCADE;")
    
    # 1. Create a Note
    note_id = "note_123"
    workspace_id = "ws_456"
    cur.execute("""
        INSERT INTO notes (id, workspace_id, title, status)
        VALUES (%s, %s, %s, %s)
    """, (note_id, workspace_id, "Meeting Notes - Q1 Strategy", "structured"))

    # 2. Add Spans (Matching PDF Example)
    spans_data = [
        ("span_1", "Strategic initiatives for Q1: Launch new product feature by March 15. Expand to European market with focus on Germany and France.", 47),
        ("span_2", "Hire 5 senior engineers to support growth. Key assumptions: Budget approved by board. Engineering capacity available.", 38),
        ("span_3", "Market research completed by January. Open questions: What's our go-to-market timeline? Do we have regulatory approval for EU?", 45)
    ]

    for s_id, text, tokens in spans_data:
        # Mock embedding (384 dim)
        mock_embedding = [random.random() for _ in range(384)]
        embedding_str = "[" + ",".join(map(str, mock_embedding)) + "]"
        
        cur.execute("""
            INSERT INTO spans (id, note_id, text, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
        """, (s_id, note_id, text, tokens, embedding_str))

    # 3. Add Objects (Idea, Assumption, Question)
    objects_data = [
        ("obj_001", "Idea", "Launch new product feature by March 15", 0.95),
        ("obj_002", "Idea", "Expand to European market with focus on Germany and France", 0.92),
        ("obj_003", "Assumption", "Budget approved by board", 0.88),
        ("obj_004", "Question", "What's our go-to-market timeline?", 0.98),
        ("obj_078", "Claim", "Physical activity shows no significant effect on mood disorders", 0.92), # For contradiction demo
        ("obj_010", "Claim", "Exercise improves mental health and reduces depression", 0.95)   # For contradiction demo
    ]

    for o_id, o_type, text, conf in objects_data:
        cur.execute("""
            INSERT INTO objects (id, workspace_id, type, canonical_text, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """, (o_id, workspace_id, o_type, text, conf))

    # 4. Add Links (Relationships)
    links_data = [
        ("link_001", "obj_003", "obj_001", "DependsOn", 0.85), # Product Launch depends on Budget
        ("link_002", "obj_004", "obj_001", "RefersTo", 0.90),  # Question refers to Launch
        ("link_500", "obj_010", "obj_078", "Contradicts", 0.92) # Contradiction example
    ]

    for l_id, src, dst, l_type, conf in links_data:
        cur.execute("""
            INSERT INTO links (id, workspace_id, src_object_id, dst_object_id, type, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (l_id, workspace_id, src, dst, l_type, conf))

    # 5. Add Insight (Contradiction)
    insight_payload = json.dumps({
        "claim1_text": "Exercise improves mental health...",
        "claim2_text": "Physical activity shows no significant effect...",
        "explanation": "Direct contradiction regarding exercise benefit."
    })
    
    cur.execute("""
        INSERT INTO insights (id, workspace_id, type, severity, payload)
        VALUES (%s, %s, %s, %s, %s)
    """, ("insight_001", workspace_id, "contradiction", "high", insight_payload))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Seed Complete! Dummy data inserted.")

if __name__ == "__main__":
    run_seed()
