import os
import sys
import time
import textwrap
import psycopg2

from ml.extraction_tasks import extract_text_task, chunk_text_task
from backend.embedding_pipeline import EmbeddingPipeline
from ml.extraction import LLMExtractor
from ml.db import init_db

# ANSI Escape codes for pretty terminal colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

DB_CONN = "dbname=note_agent user=postgres host=localhost"

# Check for API keys needed for Stage 4
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
if not API_KEY:
    print(f"{Colors.FAIL}{Colors.BOLD}⚠ CRITICAL: You must export OPENAI_API_KEY or GROQ_API_KEY to run the Stage 4 demo!{Colors.ENDC}")
    sys.exit(1)

def print_step(title, desc):
    print(f"\n{Colors.HEADER}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}► {title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"{Colors.OKCYAN}{desc}{Colors.ENDC}\n")
    time.sleep(2)

def prepare_demo_data():
    os.makedirs("/tmp/noteagent", exist_ok=True)
    sample_path = "/tmp/noteagent/demo_note.txt"
    demo_text = """
Meeting Notes - Q1 Strategy Launch
Date: March 1st

Strategic initiatives for Q1:
1. Launch new Quantum Engine product by March 15.
2. Expand into the European market, focusing strongly on Germany and France.
3. Hire 5 senior engineers to support our backend growth.

Key assumptions:
- The Q1 Budget is approved by the board of directors.
- We have the required engineering capacity available.
- Market research in the EU is already completed by January.

Open questions:
- What is our exact go-to-market timeline?
- Do we have all the regulatory approvals needed for EU expansion?
"""
    with open(sample_path, "w") as f:
        f.write(demo_text.strip())

    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()
    # Cleanup previous runs
    cur.execute("DELETE FROM spans WHERE note_id = 'demo_note_1'")
    cur.execute("DELETE FROM notes WHERE id = 'demo_note_1'")
    
    cur.execute(
        "INSERT INTO notes (id, title, status, workspace_id) VALUES ('demo_note_1', 'Demo Note', 'uploaded', 'ws_demo') ON CONFLICT DO NOTHING",
        ()
    )
    conn.commit()
    conn.close()

    # Need to also seed sqlite logic for the extraction task
    os.environ["NOTE_AGENT_DB_PATH"] = "/tmp/noteagent/test.db"
    from ml.db import init_db, _connect
    init_db()
    with _connect() as sqlite_conn:
        sqlite_conn.execute("INSERT OR REPLACE INTO notes (id, file_path, status, mime_type) VALUES (1, ?, 'uploaded', 'text/plain')", (sample_path,))
        sqlite_conn.commit()
    
    return 1, sample_path, demo_text.strip()

def run_demo():
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Starting Notes Agent ML Pipeline Demo{Colors.ENDC}\n")
    
    note_id, filepath, raw_text = prepare_demo_data()
    print(f"{Colors.BOLD}Processing Note:{Colors.ENDC} {filepath}")
    print(f"{Colors.BOLD}Raw Input Document:{Colors.ENDC}")
    print("--------------------------------------------------")
    print(raw_text)
    print("--------------------------------------------------")
    
    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 1 (Text Extraction)...{Colors.ENDC}")
    
    # STAGE 1
    print_step("STAGE 1: Text Extraction", "Taking the raw document and normalizing the text.")
    print("Calling: extract_text_task()...")
    time.sleep(1)
    
    extract_text_task(note_id)
    
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()
    cur.execute("SELECT status FROM notes WHERE id = 'demo_note_1'")
    status = cur.fetchone()[0]
    
    import sqlite3
    with sqlite3.connect("/tmp/noteagent/test.db") as s_conn:
        s_conn.row_factory = sqlite3.Row
        cleaned_txt = s_conn.execute("SELECT cleaned_text FROM notes WHERE id = 1").fetchone()["cleaned_text"]
    print(f"{Colors.OKGREEN}✓ Extraction complete. Status updated to '{status}'.{Colors.ENDC}")
    
    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 2 (Token Chunking)...{Colors.ENDC}")
    
    # STAGE 2
    print_step("STAGE 2: Sliding Window Chunking", "Breaking the normalized text into overlapping sentences and verifying token counts.")
    print("Calling: chunk_text_task()...")
    time.sleep(1)
    
    num_spans = chunk_text_task(note_id, window_size=50, overlap=10, min_tokens=20)
    
    str_note_id = str(note_id)
    cur.execute("SELECT token_count, text FROM spans WHERE note_id = %s ORDER BY start_char", (str_note_id,))
    spans = cur.fetchall()
    print(f"{Colors.OKGREEN}✓ Created {num_spans} chunks.{Colors.ENDC}\n")
    
    for idx, (tokens, text) in enumerate(spans[:3]):
        print(f"  {Colors.BOLD}[Chunk {idx:02d}] ({tokens} tokens){Colors.ENDC}: {textwrap.shorten(text, width=80)}")
    
    if len(spans) > 3:
        print("  ... (more truncated)")

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 3 (Vector Embedding)...{Colors.ENDC}")

    # STAGE 3
    print_step("STAGE 3: Vector Embeddings", "Converting the text chunks into 384-dimensional math vectors via miniLM.")
    print("Calling: EmbeddingPipeline.embed_spans_task()...")
    time.sleep(1)
    
    embedder = EmbeddingPipeline(DB_CONN)
    embedder.embed_spans_task(note_id)
    
    cur.execute("SELECT embedding IS NOT NULL FROM spans WHERE note_id = %s LIMIT 3", (str_note_id,))
    emb_results = cur.fetchall()
    
    print(f"{Colors.OKGREEN}✓ pgvector embedding HNSW index populated.{Colors.ENDC}\n")
    for idx, (is_embedded,) in enumerate(emb_results):
        print(f"  {Colors.BOLD}[Chunk {idx:02d}]{Colors.ENDC} has vector: {is_embedded}")

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 4 (Structured LLM Extraction)...{Colors.ENDC}")
    
    # STAGE 4
    print_step("STAGE 4: Structured Extraction", "Using LLMExtractor to map text objects into a Knowledge Graph (Claims, Ideas, Questions, Tasks).")
    print("Calling: LLMExtractor.extract()...")
    time.sleep(1)
    
    extractor = LLMExtractor(verbose=False)
    result = extractor.extract(cleaned_txt, note_id=str(note_id))
    
    print(f"\n{Colors.OKGREEN}✓ Extracted {len(result.objects)} objects and {len(result.links)} relationships.{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}EXTRACTED KNOWLEDGE NODES:{Colors.ENDC}")
    for obj in result.objects:
        color = Colors.OKBLUE if obj.type == 'Idea' else Colors.WARNING if obj.type == 'Question' else Colors.ENDC
        print(f"  • [{color}{obj.type}{Colors.ENDC}] {obj.canonical_text} (Conf: {obj.confidence})")
        
    if result.links:
        print(f"\n{Colors.BOLD}EXTRACTED RELATIONSHIPS (EDGES):{Colors.ENDC}")
        for link in result.links:
            print(f"  • {link.source_id} {Colors.OKCYAN}--[{link.type}]-->{Colors.ENDC} {link.target_id}")
    else:
        print(f"\n{Colors.BOLD}No graph relationships detected in this small text block.{Colors.ENDC}")
        
    # --- ADD TO DATABASE FOR DEMO ---
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()
    cur.execute("DELETE FROM links WHERE workspace_id = 'ws_demo'")
    cur.execute("DELETE FROM objects WHERE workspace_id = 'ws_demo'")
    
    cur.execute("SELECT id FROM spans WHERE note_id = 'demo_note_1' LIMIT 1")
    span_res = cur.fetchone()
    valid_span_id = span_res[0] if span_res else None

    for obj in result.objects:
        cur.execute(
            "INSERT INTO objects (id, type, canonical_text, confidence, status, workspace_id) VALUES (%s, %s, %s, %s, 'active', 'ws_demo') ON CONFLICT DO NOTHING",
            (obj.id, obj.type, obj.canonical_text, obj.confidence)
        )
        
    for i, link in enumerate(result.links):
        cur.execute(
            "INSERT INTO links (id, src_object_id, dst_object_id, type, evidence_span_id, confidence, workspace_id) VALUES (%s, %s, %s, %s, %s, %s, 'ws_demo') ON CONFLICT DO NOTHING",
            (f"link_demo_{i}", link.source_id, link.target_id, link.type, valid_span_id, link.confidence)
        )
    conn.commit()
    conn.close()

    print(f"\n{Colors.HEADER}{Colors.BOLD}=== DEMO COMPLETE ==={Colors.ENDC}\n")
    print(f"{Colors.OKCYAN}The objects and links have been saved to Postgres! Run the following to check:")
    print(f"  psql -d note_agent -c \"SELECT id, type, canonical_text FROM objects WHERE workspace_id = 'ws_demo';\"")
    print(f"  psql -d note_agent -c \"SELECT src_object_id, type, dst_object_id FROM links WHERE workspace_id = 'ws_demo';\"{Colors.ENDC}\n")

if __name__ == "__main__":
    run_demo()
