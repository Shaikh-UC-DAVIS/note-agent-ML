import os
import sys
import time
import json
import textwrap
import psycopg2
from datetime import datetime, timezone, timedelta

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

DB_CONN = "dbname=note_agent user=postgres password=postgres host=localhost"

# Check for API keys needed for Stage 4
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
if not API_KEY:
    print(f"{Colors.FAIL}{Colors.BOLD}! CRITICAL: You must export OPENAI_API_KEY or GROQ_API_KEY to run the Stage 4 demo!{Colors.ENDC}")
    sys.exit(1)

def print_step(title, desc):
    print(f"\n{Colors.HEADER}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}> {title}{Colors.ENDC}")
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

Follow-up Notes - March 5th Update:
- CORRECTION: The Q1 Budget has NOT been approved. The board rejected the initial proposal and requested a revised budget by March 20th.
- CORRECTION: Engineering capacity is NOT available. The team is fully committed to existing projects through Q2 and cannot support new hires onboarding.

Engineering Sync Notes - March 6th:
- We need to ship the Quantum Engine by mid-March.
- Plan to onboard 5 additional backend engineers to grow the team.
- EU expansion should prioritize Germany and France as key markets.
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
    print(f"{Colors.OKGREEN}+ Extraction complete. Status updated to '{status}'.{Colors.ENDC}")
    
    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 2 (Token Chunking)...{Colors.ENDC}")
    
    # STAGE 2
    print_step("STAGE 2: Sliding Window Chunking", "Breaking the normalized text into overlapping sentences and verifying token counts.")
    print("Calling: chunk_text_task()...")
    time.sleep(1)
    
    num_spans = chunk_text_task(note_id, window_size=50, overlap=10, min_tokens=20)

    # Bridge spans from SQLite → PostgreSQL so EmbeddingPipeline can find them
    import sqlite3
    with sqlite3.connect("/tmp/noteagent/test.db") as s_conn:
        s_conn.row_factory = sqlite3.Row
        sqlite_spans = s_conn.execute(
            "SELECT id, text, token_count, start_char, end_char FROM spans WHERE note_id = ?", (note_id,)
        ).fetchall()

    cur.execute("DELETE FROM spans WHERE note_id = 'demo_note_1'")
    for s in sqlite_spans:
        cur.execute(
            "INSERT INTO spans (id, note_id, start_char, end_char, text, token_count) VALUES (%s, 'demo_note_1', %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (f"span_demo_{s['id']}", s['start_char'], s['end_char'], s['text'], s['token_count'])
        )
    conn.commit()

    str_note_id = 'demo_note_1'
    cur.execute("SELECT token_count, text FROM spans WHERE note_id = %s ORDER BY start_char", (str_note_id,))
    spans = cur.fetchall()
    print(f"{Colors.OKGREEN}+ Created {num_spans} chunks.{Colors.ENDC}\n")

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
    embedder.embed_spans_task(str_note_id)
    
    cur.execute("SELECT embedding IS NOT NULL FROM spans WHERE note_id = %s LIMIT 3", (str_note_id,))
    emb_results = cur.fetchall()
    
    print(f"{Colors.OKGREEN}+ pgvector embedding HNSW index populated.{Colors.ENDC}\n")
    for idx, (is_embedded,) in enumerate(emb_results):
        print(f"  {Colors.BOLD}[Chunk {idx:02d}]{Colors.ENDC} has vector: {is_embedded}")

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 4 (Structured LLM Extraction)...{Colors.ENDC}")
    
    # STAGE 4
    print_step("STAGE 4: Structured Extraction", "Using LLMExtractor to map text objects into a Knowledge Graph (Claims, Ideas, Questions, Tasks).")
    print("Calling: LLMExtractor.extract()...")
    time.sleep(1)
    
    extractor = LLMExtractor(verbose=False)
    result = extractor.extract(cleaned_txt, note_id=str(note_id))
    
    print(f"\n{Colors.OKGREEN}+ Extracted {len(result.objects)} objects and {len(result.links)} relationships.{Colors.ENDC}\n")
    
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

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 5 (Entity Resolution)...{Colors.ENDC}")

    # STAGE 5
    print_step("STAGE 5: Entity Resolution",
               "Comparing new objects against existing Knowledge Graph nodes to detect duplicates.")

    from ml.entity_resolution import EntityResolver
    resolver = EntityResolver(DB_CONN)
    new_obj_ids = [obj.id for obj in result.objects]

    # Seed a pre-existing object with identical canonical text to the first extracted
    # object so the demo reliably demonstrates auto-merge (similarity = 1.0 → merge).
    if result.objects:
        seed_obj = result.objects[0]
        seed_embedding = resolver._embed_texts([seed_obj.canonical_text])[0]
        vec_str = "[" + ",".join(map(str, seed_embedding)) + "]"
        seed_conn = psycopg2.connect(DB_CONN)
        seed_cur = seed_conn.cursor()
        seed_cur.execute("DELETE FROM objects WHERE id = 'pre_existing_1'")
        seed_cur.execute(
            "INSERT INTO objects (id, type, canonical_text, confidence, status, workspace_id, embedding) "
            "VALUES ('pre_existing_1', %s, %s, 0.9, 'active', 'ws_demo', %s::vector)",
            (seed_obj.type, seed_obj.canonical_text, vec_str)
        )
        seed_conn.commit()
        seed_conn.close()
        print(f"  Pre-existing seed: [{seed_obj.type}] \"{seed_obj.canonical_text[:70]}\"")

    print("Calling: EntityResolver.resolve_entities_task()...")
    stats = resolver.resolve_entities_task(new_obj_ids, workspace_id='ws_demo')

    print(f"{Colors.OKGREEN}+ Entity Resolution complete.{Colors.ENDC}")
    print(f"  Merged:    {stats['merged']}")
    print(f"  Flagged:   {stats['flagged']}")
    print(f"  Unchanged: {stats['unchanged']}")

    print(f"\n{Colors.OKCYAN}Verify in psql:")
    print(f"  SELECT id, status FROM objects WHERE workspace_id = 'ws_demo';")
    print(f"  SELECT type, src_object_id, dst_object_id FROM links WHERE type = 'SameAs';")
    print(f"  SELECT type, severity, payload FROM insights WHERE type = 'consolidation_opportunity';{Colors.ENDC}")

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 6 (Hybrid Search)...{Colors.ENDC}")

    # STAGE 6
    print_step("STAGE 6: Hybrid Search", "Combining vector similarity and keyword matching via Reciprocal Rank Fusion.")

    from ml.search import HybridSearchEngine
    from ml.graph import KnowledgeGraph

    # Build knowledge graph from extracted objects
    kg = KnowledgeGraph()
    kg.add_objects(result.objects)
    kg.add_links(result.links)

    # Build search engine in in-memory mode (no storage adapter needed)
    search_engine = HybridSearchEngine(embedding_generator=embedder, graph=kg)

    # Load spans from SQLite (chunk_text_task writes to SQLite, not Postgres)
    import sqlite3 as _sqlite3
    with _sqlite3.connect("/tmp/noteagent/test.db") as s_conn:
        s_conn.row_factory = _sqlite3.Row
        rows = s_conn.execute(
            "SELECT id, text, token_count FROM spans WHERE note_id = ?", (note_id,)
        ).fetchall()
    spans_data = [(row["id"], row["text"], row["token_count"]) for row in rows]

    print(f"Indexing {len(spans_data)} span(s) into search engine...")
    for span_id, text, token_count in spans_data:
        vec = embedder.model.encode(text).tolist()
        search_engine.index_chunk(span_id, text, vec, token_count or 0)

    demo_query = "When is the Quantum Engine product launching?"
    print(f"\nQuery: \"{demo_query}\"")
    search_results = search_engine.search(demo_query, top_k=3)

    print(f"\n{Colors.OKGREEN}+ Top {len(search_results)} result(s):{Colors.ENDC}")
    for i, r in enumerate(search_results):
        print(f"  {Colors.BOLD}[{i+1}] score={r.score:.4f}  source={r.source}{Colors.ENDC}")
        print(f"      {textwrap.shorten(r.text, width=90)}")

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 7 (Contradiction Detection)...{Colors.ENDC}")

    # STAGE 7
    print_step("STAGE 7: Contradiction Detection", "Scanning the Knowledge Graph for conflicting claims.")

    from ml.intelligence import IntelligenceLayer

    intel = IntelligenceLayer(kg)
    contradictions = intel.detect_contradictions()

    if contradictions:
        print(f"\n{Colors.OKGREEN}+ Found {len(contradictions)} contradiction(s):{Colors.ENDC}")
        for c in contradictions:
            print(f"  • [{Colors.FAIL}HIGH{Colors.ENDC}] \"{c['source_text'][:60]}\"")
            print(f"    ↔ \"{c['target_text'][:60]}\"")
    else:
        print(f"\n{Colors.OKGREEN}+ No contradictions detected.{Colors.ENDC}")

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 8 (Stale Thread Detection)...{Colors.ENDC}")

    # STAGE 8
    print_step("STAGE 8: Stale Thread Detection", "Finding open Questions, Tasks, and Ideas that haven't been addressed.")

    # Simulate objects being 60 days old so threads appear stale in the demo
    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    for node_id in kg.graph.nodes:
        kg.graph.nodes[node_id]['created_at'] = old_date

    intel_stale = IntelligenceLayer(kg)
    stale_insights = intel_stale.detect_stale_threads()

    if stale_insights:
        print(f"\n{Colors.OKGREEN}+ Found {len(stale_insights)} stale thread(s):{Colors.ENDC}")
        for ins in stale_insights:
            p = ins['payload']
            color = Colors.FAIL if ins['severity'] == 'high' else Colors.WARNING
            print(f"  • [{color}{ins['severity'].upper()}{Colors.ENDC}] [{p['object_type']}] \"{p['object_text'][:70]}\" ({p['age_days']}d old)")
    else:
        print(f"\n{Colors.OKGREEN}+ No stale threads detected.{Colors.ENDC}")

    input(f"\n{Colors.WARNING}Press [ENTER] to execute Stage 9 (Consolidation Detection)...{Colors.ENDC}")

    # STAGE 9
    print_step("STAGE 9: Consolidation Detection", "Reviewing near-duplicate entities flagged during Entity Resolution (Stage 5).")

    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()
    cur.execute(
        "SELECT severity, payload FROM insights WHERE type = 'consolidation_opportunity' AND workspace_id = 'ws_demo'"
    )
    consolidation_rows = cur.fetchall()
    conn.close()

    if consolidation_rows:
        print(f"\n{Colors.OKGREEN}+ Found {len(consolidation_rows)} consolidation opportunity(ies):{Colors.ENDC}")
        for severity, payload in consolidation_rows:
            p = json.loads(payload) if isinstance(payload, str) else payload
            color = Colors.FAIL if severity == 'high' else Colors.WARNING
            print(f"  • [{color}{severity.upper()}{Colors.ENDC}] similarity={p.get('similarity', '?')}")
            print(f"    {p.get('src_id', '?')[:16]}... ↔ {p.get('dst_id', '?')[:16]}...")
            print(f"    Reason: {p.get('reason', '')}")
    else:
        print(f"\n  No consolidation opportunities flagged — all entities were auto-merged or unique.")

    print(f"\n{Colors.HEADER}{Colors.BOLD}=== DEMO COMPLETE ==={Colors.ENDC}\n")

if __name__ == "__main__":
    run_demo()
