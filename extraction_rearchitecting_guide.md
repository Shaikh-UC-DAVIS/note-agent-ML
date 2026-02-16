# Knowledge Extraction System: Technical Guide
**Subject**: Rearchitecting `ml/extraction.py` from Heuristic Mock to Probabilistic LLM Engine

## 1. Overview
The Note Agent's extraction layer has been upgraded from a static, rule-based mock to a dynamic, LLM-powered engine. This allows the system to extract structured knowledge (Claims, Ideas, Links) from arbitrary text without needing pre-defined keywords.

## 2. Core Architectural Changes

### From Mock to Model
- **Old Heuristic**: Scanned for strings like "earth" or "gravity" to return hardcoded objects.
- **New Engine**: Utilizes **Groq's Llama 3.3 70B** model to perform deep semantic analysis of the input text.

### Provider-Agnostic Client
We utilize the `openai` Python library but point it at **Groq's API** (`https://api.groq.com/openai/v1`).
- **Benefit**: This architecture avoids vendor lock-in. We can switch between OpenAI, Gemini, or local models just by updating the `base_url` in the `LLMExtractor` constructor.

### Environmental Security
API credentials have been moved to a `.env` file (loaded via `python-dotenv`).
- **Standard**: Prevents accidental exposure of private keys in source control.
- **Gitignore**: Added `.env` to `.gitignore` to ensure security during research collaboration.

---

## 3. High-Level AI Concepts (The "Meetings Speak")

### Zero-Shot Extraction
We utilize **Zero-Shot Prompting**. The model is given a complex instruction set but zero specific examples. This demonstrates the high reasoning capability of the Llama 3.3 backbone.
> *Future potential: Move to "Few-Shot" by adding examples to the System Prompt if accuracy needs to be tuned for specific academic domains.*

### Role-Based Privilege (ChatML)
The request is split into **System** and **User** roles:
- **System Role**: Defines the "Laws of the Physics" (Ontology). It has higher internal attention weight and protects against "Prompt Injection."
- **User Role**: Contains the volatile data (the user's notes) to be processed.

### Deterministic Parameters
- **Temperature (0.2)**: Set low to minimize "creativity" and maximize **reproducibility**. For data extraction, we need the same text to result in the same graph every time.
- **JSON Mode**: Forced via the API (`response_format={"type": "json_object"}`). This ensures the model treats the output like a data structure, not a conversation.

---

## 4. Engineering Implementation

### The Knowledge Ontology
We defined a strict schema for the AI to follow:
- **Objects**: Idea, Claim, Assumption, Question, Task, Evidence, Definition.
- **Links**: Supports, Contradicts, Refines, DependsOn, SameAs, Causes.

### Structured Validation Layer
We use **Pydantic** (`ExtractionResult`) as a "Quality Control" gate.
1. The LLM generates a JSON string.
2. `json.loads` converts it to a Python dict.
3. `ExtractionResult(**parsed)` validates it.
- **Outcome**: If the LLM generates an invalid type or misses a field, the system catches it gracefully instead of polluting the Knowledge Graph with "dirty" data.

### Sliding-Window Ingestion
To handle documents larger than the LLM's **Context Window**, the system uses a sliding-window strategy to process text in chunks, ensuring no information is lost due to memory constraints.

---

## 5. Verification & Testing
Two tools were created to verify these changes:
1. `test_extraction.py`: A standalone smoke test to verify API connectivity and extraction logic.
2. `demo.py`: An end-to-end visualization of the entire pipeline (Chunking -> Embedding -> Extraction -> Graph Construction).

## 6. Research Impact
By converting unstructured notes into a **Directed Acyclic Graph (DAG)** of claims and evidence, the system enables advanced downstream tasks:
- **Contradiction Detection**: Identifying conflicting claims across different notes.
- **Graph Centrality**: Finding "Core Concepts" that link many disparate ideas together.
- **Stale Thread Analysis**: Finding questions that remain unanswered in the graph.
