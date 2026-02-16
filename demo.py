import os
import sys
import json

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from ml.ingestion import TextIngestion, EmbeddingGenerator
from ml.extraction import LLMExtractor
from ml.graph import KnowledgeGraph
from ml.search import HybridSearchEngine
from ml.intelligence import IntelligenceLayer

# Import the Postgres storage layer
from backend.postgres_storage import PostgresMetadataStorage

def main():
    print("\n" + "="*60)
    print(" NOTE AGENT: DATA TRANSFORMATION PIPELINE VISUALIZATION")
    print("="*60 + "\n")

    # 1. Setup
    print("[INIT] Loading Components...")
    ingestion = TextIngestion()
    
    # Try loading real embedder, else use a mock for visualization
    try:
        embedder = EmbeddingGenerator()
        print("   -> Embedding Model: Loaded (sentence-transformers)")
    except Exception:
        print("   -> Embedding Model: Not found. Using MOCK for visualization.")
        class MockEmbedder:
            def generate_embeddings(self, chunks):
                # Return random 3-dim vectors for demo
                import random
                return [[random.random() for _ in range(3)] for _ in chunks]
            class Model:
                def encode(self, query):
                    import random
                    return [random.random() for _ in range(3)]
            model = Model()
        embedder = MockEmbedder()
        
    extractor = LLMExtractor()
    graph = KnowledgeGraph()

    # Initialize PostgresMetadataStorage
    conn_string = "dbname=note_agent user=postgres password=postgres host=localhost"
    storage = PostgresMetadataStorage(conn_string)

    # Pass storage to HybridSearchEngine
    search_engine = HybridSearchEngine(embedding_generator=embedder, graph=graph, storage=storage)
    intelligence = IntelligenceLayer(graph)

    # 2. Raw Input
    raw_text = """The earth is round. This is a claim supported by scientific evidence.
However, some people believe the earth is flat. This contradicts the scientific consensus.
Key Idea: Gravity pulls everything towards the center of mass."""
    
    print(f"\n[STEP 1] RAW INPUT TEXT")
    print("-" * 30)
    print(f"'{raw_text}'")
    print("-" * 30)

    # 3. Chunking
    print(f"\n[STEP 2] CHUNKING (Text -> Chunks)")
    chunks = ingestion.chunk_text(raw_text, window_size=20, overlap=5) # Small window to force multiple chunks for demo
    print(f"   -> Strategy: Sliding Window (size=20 tokens, overlap=5)")
    print(f"   -> Result: {len(chunks)} Chunks Generated")
    for i, chunk in enumerate(chunks):
        print(f"      [Chunk {i}] (Tokens: {chunk.token_count}): \"{chunk.text.replace(chr(10), ' ')}\"")

    # 4. Embedding
    print(f"\n[STEP 3] EMBEDDING (Chunks -> Vectors)")
    embeddings = embedder.generate_embeddings(chunks)
    print(f"   -> Result: {len(embeddings)} Vectors Generated")

    # Save chunks + embeddings to Postgres
    for i, vec in enumerate(embeddings):
        chunk = chunks[i]
        search_engine.index_chunk(
            f"chunk-{i}",       # chunk id
            chunks[i].text,     # text
            vec,                # vector
            chunks[i].token_count  # token_count if function supports
        )
        # Show snippet
        vec_preview = ", ".join([f"{x:.4f}" for x in vec[:3]])
        print(f"      [Vector {i}] [{vec_preview}, ...]")

    # 5. Structured Extraction
    print(f"\n[STEP 4] STRUCTURED EXTRACTION (Text -> Objects)")
    print(f"   -> Model: LLM (Mocked)")
    extraction_result = extractor.extract(raw_text)
    
    print(f"   -> Extracted Objects:")
    for obj in extraction_result.objects:
        print(f"      - [{obj.type}] {obj.id}: \"{obj.canonical_text}\" (Conf: {obj.confidence})")
        
    print(f"   -> Extracted Links:")
    for link in extraction_result.links:
        print(f"      - {link.source_id} --[{link.type}]--> {link.target_id}")

    # 6. Knowledge Graph
    print(f"\n[STEP 5] KNOWLEDGE GRAPH CONSTRUCTION (Objects -> Graph)")
    graph.add_objects(extraction_result.objects)
    graph.add_links(extraction_result.links)
    print(f"   -> Graph Stats: {graph.graph.number_of_nodes()} Nodes, {graph.graph.number_of_edges()} Edges")
    print(f"   -> Nodes: {list(graph.graph.nodes())}")

    # 7. Search
    print(f"\n[STEP 6] HYBRID SEARCH (Query -> Ranked Segments)")
    query = "earth shape"
    print(f"   -> Query: '{query}'")
    results = search_engine.search(query)
    for i, res in enumerate(results):
        print(f"      {i+1}. [{res['id'].upper()}] Score: {res['score']:.4f} | \"{res['text'].replace(chr(10), ' ')[:40]}...\"")

    # 8. Intelligence
    print(f"\n[STEP 7] INTELLIGENCE LAYER (Graph -> Insights)")
    insights = intelligence.generate_insights()
    for insight in insights:
        print(f"   -> [INSIGHT] {insight['type']}: {insight['text']} ({insight['message']})")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
