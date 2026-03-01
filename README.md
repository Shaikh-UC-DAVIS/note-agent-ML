# Note Agent: Machine Learning Pipeline

The ML pipeline transforms raw documents (PDFs, DOCX) into a structured Knowledge Graph, enabling semantic search and intelligent insights (contradiction detection, consolidation).

## 🚀 Quick Start for Developers

### 1. Prerequisites
- Python 3.10+
- PostgreSQL 16 with `pgvector` extension
- Docker (optional, for running Postgres easily)

### 2. Environment Setup
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup
You need a running Postgres instance.
```bash
# Option A: Docker (Recommended)
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg16

# Option B: Local Install
# Ensure you have 'postgres' user created and 'pgvector' extension installed
```

### 4. Initialize & Seed Data
We rely on a set of Python scripts to manage the database state.
```bash
# Create the database 'note_agent'
python scripts/create_db.py

# Apply schema and seed with 'Strategic Q1' dummy data
python scripts/seed_db.py
```

## 🛠️ Developer Tools (`scripts/`)

| Script | Description |
| :--- | :--- |
| `create_db.py` | Creates the `note_agent` database if missing. |
| `seed_db.py` | Applies `backend/schema.sql` and populates dummy data (Notes, Spans, Objects). |
| `inspect_db.py` | CLI tool to list tables and view rows as JSON. |
| `check_data.py` | Runs a health check and prints counts of all entities. |

**Usage Examples:**
```bash
# Check if data loaded correctly
python scripts/check_data.py

# Inspect specific table content
python scripts/inspect_db.py objects --limit 5
```

## 📂 Project Structure

- **`ml/`**: Core logic for the pipeline.
    - `extraction_tasks.py`: Text extraction and chunking tasks.
    - `extraction.py`: LLM client for structured data.
    - `graph.py`: NetworkX graph operations.
    - `search.py`: Hybrid search (Vector + Keyword).
- **`backend/`**: Database interactions.
    - `schema.sql`: Source of truth for DB structure.
    - `postgres_storage.py`: Storage class for artifacts.
- **`scripts/`**: DevOps and utility scripts.
