"""
HTTP controller for the ML service.

Wraps the existing ml/ and backend/ modules as stateless HTTP endpoints so
consumers (e.g. the Notes backend) can call them over the network instead of
importing the Python packages directly.

Run:
    pip install -r api/requirements.txt
    # from repo root:
    uvicorn api.server:app --host 0.0.0.0 --port 9000

No existing files in ml/ or backend/ are modified.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Repo path setup so `ml.*` / `backend.*` imports resolve ────────────────
import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Auth ────────────────────────────────────────────────────────────────────
INTERNAL_KEY = os.getenv("ML_INTERNAL_KEY", "")


def require_key(x_internal_key: Optional[str] = Header(default=None)):
    if INTERNAL_KEY and x_internal_key != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="invalid internal key")
    return True


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Note Agent ML Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy singletons (heavy models load once per process) ────────────────────
_EMBED_MODEL = None
_EMBED_MODEL_LOCK = threading.Lock()
_NLP = None
_NLP_LOCK = threading.Lock()
_WHISPER = None
_WHISPER_LOCK = threading.Lock()


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _EMBED_MODEL_LOCK:
            if _EMBED_MODEL is None:
                from sentence_transformers import SentenceTransformer
                name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                _EMBED_MODEL = SentenceTransformer(name)
    return _EMBED_MODEL


_LLM_CLIENT = None
_LLM_MODEL = None
_LLM_LOCK = threading.Lock()


def _get_llm():
    """Build an OpenAI-compatible client pointed at Groq or OpenAI, matching
    the selection logic used by ml/extraction.py."""
    global _LLM_CLIENT, _LLM_MODEL
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT, _LLM_MODEL
    with _LLM_LOCK:
        if _LLM_CLIENT is not None:
            return _LLM_CLIENT, _LLM_MODEL
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        key = openai_key or groq_key
        if not key:
            raise HTTPException(status_code=503, detail="no LLM API key configured on ML service")
        if key.startswith("sk-"):
            base_url = "https://api.openai.com/v1"
            _LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            base_url = "https://api.groq.com/openai/v1"
            _LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        _LLM_CLIENT = OpenAI(api_key=key, base_url=base_url)
        return _LLM_CLIENT, _LLM_MODEL


def _get_spacy():
    global _NLP
    if _NLP is None:
        with _NLP_LOCK:
            if _NLP is None:
                try:
                    import spacy
                    _NLP = spacy.load(
                        "en_core_web_sm",
                        exclude=["ner", "tagger", "lemmatizer"],
                    )
                except Exception:
                    _NLP = False  # flag as unavailable
    return _NLP or None


# ── Schemas ─────────────────────────────────────────────────────────────────
ObjectType = Literal["Idea", "Claim", "Assumption", "Question", "Task", "Evidence", "Definition"]
LinkType = Literal["Supports", "Contradicts", "Refines", "DependsOn", "SameAs", "Causes"]


class Span(BaseModel):
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    text: str


class ChunkReq(BaseModel):
    text: str
    window_tokens: int = 400
    min_tokens: int = 80


class ChunkResp(BaseModel):
    spans: List[Span]


class EmbedReq(BaseModel):
    texts: List[str]


class EmbedResp(BaseModel):
    vectors: List[List[float]]
    model: str
    dim: int


class ExtractReq(BaseModel):
    text: str
    note_id: Optional[str] = None
    workspace_id: Optional[str] = None


class ObjectOut(BaseModel):
    id: str
    type: ObjectType
    canonical_text: str
    confidence: float


class LinkOut(BaseModel):
    source_id: str
    target_id: str
    type: LinkType
    confidence: float


class MentionOut(BaseModel):
    object_id: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class ExtractResp(BaseModel):
    objects: List[ObjectOut]
    links: List[LinkOut]
    mentions: List[MentionOut]


class GraphInput(BaseModel):
    objects: List[ObjectOut]
    links: List[LinkOut] = []


class ContradictionsResp(BaseModel):
    contradictions: List[Dict[str, Any]]


class StaleThreadsReq(BaseModel):
    objects: List[Dict[str, Any]]  # must include id, type, canonical_text, created_at (ISO)
    links: List[LinkOut] = []
    days: int = 7


class StaleThreadsResp(BaseModel):
    stale: List[Dict[str, Any]]


class ResolveCandidate(BaseModel):
    id: str
    canonical_text: str
    embedding: Optional[List[float]] = None  # 384-dim, optional


class ResolveReq(BaseModel):
    new_objects: List[ResolveCandidate]
    existing_objects: List[ResolveCandidate] = []
    auto_merge_threshold: float = 0.95
    flag_threshold: float = 0.85


class MergePair(BaseModel):
    new_id: str
    existing_id: str
    similarity: float


class ResolveResp(BaseModel):
    merges: List[MergePair]
    consolidation_candidates: List[MergePair]
    unchanged_ids: List[str]


class SearchCorpusItem(BaseModel):
    span_id: str
    text: str
    embedding: Optional[List[float]] = None


class SearchReq(BaseModel):
    query: str
    corpus: List[SearchCorpusItem]
    k: int = 10
    alpha: float = 0.5  # weight between vector (alpha) and keyword (1-alpha) RRF


class SearchHit(BaseModel):
    span_id: str
    text: str
    score: float
    source: str  # "vec" | "fts" | "both"


class SearchResp(BaseModel):
    hits: List[SearchHit]


class ChatSpan(BaseModel):
    id: str
    text: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatReq(BaseModel):
    question: str
    spans: List[ChatSpan] = []
    history: List[ChatMessage] = []
    max_tokens: int = 600
    temperature: float = 0.5


class ChatResp(BaseModel):
    answer: str
    citations: List[str]
    model: str


class ProcessReq(BaseModel):
    text: str
    note_id: Optional[str] = None
    workspace_id: Optional[str] = None
    run_extraction: bool = True
    run_insights: bool = True


class ProcessResp(BaseModel):
    spans: List[Span]
    embeddings: List[List[float]]
    objects: List[ObjectOut]
    links: List[LinkOut]
    mentions: List[MentionOut]
    contradictions: List[Dict[str, Any]]
    stale_threads: List[Dict[str, Any]]
    status: str


# ── Chunking (ported algorithm, no DB) ──────────────────────────────────────
def _chunk_text(text: str, window_tokens: int = 400, min_tokens: int = 80) -> List[Dict[str, Any]]:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    nlp = _get_spacy()

    if nlp is not None:
        doc = nlp(text)
        sentences = [
            {"text": s.text, "start_char": s.start_char, "end_char": s.end_char,
             "token_count": len(encoding.encode(s.text))}
            for s in doc.sents if s.text.strip()
        ]
    else:
        sentences = []
        for m in re.finditer(r".+?(?:[.!?](?:\s+|$)|\n+|$)", text, re.DOTALL):
            s = m.group(0).strip()
            if s:
                sentences.append({
                    "text": s,
                    "start_char": m.start(),
                    "end_char": m.end(),
                    "token_count": len(encoding.encode(s)),
                })

    if not sentences:
        return []

    prefix_tokens = [0]
    for s in sentences:
        prefix_tokens.append(prefix_tokens[-1] + s["token_count"])

    spans: List[Dict[str, Any]] = []
    start_idx = 0
    chunk_index = 0
    while start_idx < len(sentences):
        current = 0
        end_idx = start_idx
        while end_idx < len(sentences):
            nxt = current + sentences[end_idx]["token_count"]
            if nxt > window_tokens and current >= min_tokens:
                break
            current = nxt
            end_idx += 1
        if end_idx == start_idx:
            end_idx = start_idx + 1
            current = sentences[start_idx]["token_count"]

        start_char = sentences[start_idx]["start_char"]
        end_char = sentences[end_idx - 1]["end_char"]
        spans.append({
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
            "token_count": current,
            "text": text[start_char:end_char],
        })

        step_tokens = max(1, int(current * 0.9))
        target = prefix_tokens[start_idx] + step_tokens
        nxt = start_idx + 1
        while nxt < len(prefix_tokens) and prefix_tokens[nxt] < target:
            nxt += 1
        start_idx = min(nxt, len(sentences))
        chunk_index += 1

    return spans


# ── Similarity helpers ──────────────────────────────────────────────────────
def _cosine(a: List[float], b: List[float]) -> float:
    import numpy as np
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = (float(np.linalg.norm(va)) * float(np.linalg.norm(vb))) or 1e-12
    return float(va @ vb / denom)


def _ensure_vectors(items: List[ResolveCandidate]) -> List[List[float]]:
    need = [i for i in items if not i.embedding]
    if need:
        model = _get_embed_model()
        vecs = model.encode([i.canonical_text for i in need], show_progress_bar=False).tolist()
        it = iter(vecs)
        for i in items:
            if not i.embedding:
                i.embedding = next(it)
    return [i.embedding for i in items]  # type: ignore[return-value]


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "service": "note-agent-ml"}


@app.post("/ml/chunk", response_model=ChunkResp, dependencies=[Depends(require_key)])
def chunk_endpoint(req: ChunkReq):
    spans = _chunk_text(req.text, window_tokens=req.window_tokens, min_tokens=req.min_tokens)
    return {"spans": spans}


@app.post("/ml/embed", response_model=EmbedResp, dependencies=[Depends(require_key)])
def embed_endpoint(req: EmbedReq):
    if not req.texts:
        return {"vectors": [], "model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), "dim": 384}
    model = _get_embed_model()
    vecs = model.encode(req.texts, batch_size=32, show_progress_bar=False).tolist()
    return {
        "vectors": vecs,
        "model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "dim": len(vecs[0]) if vecs else 384,
    }


@app.post("/ml/extract", response_model=ExtractResp, dependencies=[Depends(require_key)])
def extract_endpoint(req: ExtractReq):
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise HTTPException(status_code=503, detail="no LLM API key configured on ML service")
    from ml.extraction import LLMExtractor
    try:
        result = LLMExtractor(verbose=False).extract(
            req.text, note_id=req.note_id or str(uuid4())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extraction failed: {e}")
    objects = [
        {"id": o.id, "type": o.type, "canonical_text": o.canonical_text, "confidence": o.confidence}
        for o in (result.objects or [])
    ]
    links = [
        {"source_id": l.source_id, "target_id": l.target_id, "type": l.type, "confidence": l.confidence}
        for l in (result.links or [])
    ]
    mentions = []
    for m in (getattr(result, "mentions", None) or []):
        mentions.append({
            "object_id": getattr(m, "object_id", None) or getattr(m, "id", ""),
            "start_char": getattr(m, "start_char", None),
            "end_char": getattr(m, "end_char", None),
        })
    return {"objects": objects, "links": links, "mentions": mentions}


@app.post("/ml/insights/contradictions", response_model=ContradictionsResp, dependencies=[Depends(require_key)])
def contradictions_endpoint(req: GraphInput):
    from ml.extraction import ExtractedObject, Link as MLLink
    from ml.graph import KnowledgeGraph
    from ml.intelligence import IntelligenceLayer

    objs = [ExtractedObject(id=o.id, type=o.type, canonical_text=o.canonical_text, confidence=o.confidence)
            for o in req.objects]
    lnks = [MLLink(source_id=l.source_id, target_id=l.target_id, type=l.type, confidence=l.confidence)
            for l in req.links]
    kg = KnowledgeGraph()
    kg.add_objects(objs)
    kg.add_links(lnks)
    layer = IntelligenceLayer(kg)
    return {"contradictions": layer.detect_contradictions()}


@app.post("/ml/insights/stale-threads", response_model=StaleThreadsResp, dependencies=[Depends(require_key)])
def stale_threads_endpoint(req: StaleThreadsReq):
    """Stateless stale-thread check.

    Re-implements the rule from ml/intelligence.py without requiring a graph
    with timestamps stored as node attrs — callers pass objects + created_at.
    """
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=req.days)
    TARGET_TYPES = {"Question", "Task", "Idea"}

    outgoing = {}
    for l in req.links:
        outgoing.setdefault(l.source_id, []).append(l.type)

    stale = []
    for o in req.objects:
        if o.get("type") not in TARGET_TYPES:
            continue
        created_raw = o.get("created_at")
        if not created_raw:
            continue
        try:
            created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created > cutoff:
            continue
        outs = outgoing.get(o["id"], [])
        resolved_by = {"Supports", "Refines", "Causes"}
        if any(t in resolved_by for t in outs):
            continue
        age_days = (now - created).days
        severity = "high" if age_days > 30 else "medium" if age_days > 14 else "low"
        stale.append({
            "id": f"insight_{uuid4().hex[:8]}",
            "object_id": o["id"],
            "type": o["type"],
            "text": o.get("canonical_text", ""),
            "age_days": age_days,
            "severity": severity,
            "payload": {"object_id": o["id"], "age_days": age_days, "text": o.get("canonical_text", "")},
        })
    return {"stale": stale}


@app.post("/ml/resolve", response_model=ResolveResp, dependencies=[Depends(require_key)])
def resolve_endpoint(req: ResolveReq):
    new_vecs = _ensure_vectors(req.new_objects)
    exist_vecs = _ensure_vectors(req.existing_objects) if req.existing_objects else []

    merges: List[MergePair] = []
    flags: List[MergePair] = []
    unchanged: List[str] = []

    for i, new_obj in enumerate(req.new_objects):
        best_id: Optional[str] = None
        best_sim: float = -1.0
        for j, ex in enumerate(req.existing_objects):
            if ex.id == new_obj.id:
                continue
            sim = _cosine(new_vecs[i], exist_vecs[j])
            if sim > best_sim:
                best_sim, best_id = sim, ex.id
        if best_id is None:
            unchanged.append(new_obj.id)
            continue
        pair = MergePair(new_id=new_obj.id, existing_id=best_id, similarity=best_sim)
        if best_sim >= req.auto_merge_threshold:
            merges.append(pair)
        elif best_sim >= req.flag_threshold:
            flags.append(pair)
        else:
            unchanged.append(new_obj.id)

    return {"merges": merges, "consolidation_candidates": flags, "unchanged_ids": unchanged}


@app.post("/ml/search", response_model=SearchResp, dependencies=[Depends(require_key)])
def search_endpoint(req: SearchReq):
    """Stateless hybrid search: caller sends the corpus, we rank and return hits.

    Combines cosine similarity over embeddings with keyword overlap, merged via
    reciprocal rank fusion (RRF). No DB required.
    """
    if not req.corpus:
        return {"hits": []}

    # Ensure all corpus items + the query have vectors
    missing = [c for c in req.corpus if not c.embedding]
    model = _get_embed_model()
    if missing:
        vecs = model.encode([c.text for c in missing], show_progress_bar=False).tolist()
        it = iter(vecs)
        for c in req.corpus:
            if not c.embedding:
                c.embedding = next(it)
    query_vec = model.encode([req.query], show_progress_bar=False).tolist()[0]

    # Vector ranking
    vec_scores = [(c.span_id, c.text, _cosine(query_vec, c.embedding or [])) for c in req.corpus]
    vec_ranked = sorted(vec_scores, key=lambda x: x[2], reverse=True)

    # Keyword ranking (simple token overlap)
    q_tokens = {t for t in re.findall(r"\w+", req.query.lower()) if len(t) > 1}
    kw_scores: List[Tuple[str, str, float]] = []
    for c in req.corpus:
        c_tokens = {t for t in re.findall(r"\w+", c.text.lower()) if len(t) > 1}
        overlap = len(q_tokens & c_tokens)
        kw_scores.append((c.span_id, c.text, float(overlap)))
    kw_ranked = sorted(kw_scores, key=lambda x: x[2], reverse=True)

    # RRF merge
    K = 60
    rrf: Dict[str, Dict[str, Any]] = {}
    for rank, (sid, text, _) in enumerate(vec_ranked):
        rrf.setdefault(sid, {"text": text, "score": 0.0, "source": set()})
        rrf[sid]["score"] += req.alpha * (1.0 / (K + rank))
        rrf[sid]["source"].add("vec")
    for rank, (sid, text, score) in enumerate(kw_ranked):
        if score == 0:
            continue
        rrf.setdefault(sid, {"text": text, "score": 0.0, "source": set()})
        rrf[sid]["score"] += (1 - req.alpha) * (1.0 / (K + rank))
        rrf[sid]["source"].add("fts")

    hits = [
        {
            "span_id": sid,
            "text": v["text"],
            "score": v["score"],
            "source": "both" if len(v["source"]) == 2 else next(iter(v["source"])),
        }
        for sid, v in rrf.items()
    ]
    hits.sort(key=lambda h: h["score"], reverse=True)
    return {"hits": hits[: req.k]}


@app.post("/ml/chat", response_model=ChatResp, dependencies=[Depends(require_key)])
def chat_endpoint(req: ChatReq):
    """Stateless RAG chat.

    Caller (backend) sends the user's question plus the spans it has already
    retrieved as context. ML builds a grounded prompt, calls the LLM, and
    returns the answer plus any cited span ids it referenced.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    client, model = _get_llm()

    context_blocks = []
    allowed_ids = set()
    for s in req.spans[:20]:  # cap context size
        sid = s.id
        allowed_ids.add(sid)
        snippet = (s.text or "").strip().replace("\r", " ")
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        context_blocks.append(f"[{sid}]\n{snippet}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no relevant notes were found)"

    system_prompt = (
        "You are the user's personal notes assistant. Answer like a smart colleague who has read all of "
        "their notes and tasks — direct, varied, never templated. Use ONLY facts from the numbered context "
        "below. Each item is tagged [N1], [N2], etc. After every concrete fact, append the matching tag.\n\n"
        "MATCH THE QUESTION — do not over-answer:\n"
        "- DIRECT factual lookup ('when X', 'who X', 'where X', 'how many X', 'what is the status of X', "
        "'what was X before/after Y'): answer in ONE short sentence. Do NOT add related context, "
        "background, or implications the user didn't ask about.\n"
        "- BROAD / synthesis question ('tell me about X', 'summarize X', 'what's going on with Y', "
        "'what does X involve'): combine facts across 2–3 items into one coherent answer. Weave them "
        "together rather than listing per-item. Surface connections (cause/effect, before/after, "
        "who-owns-what) that the items make plain.\n"
        "- LIST question ('list all', 'what are the risks', 'every step'): use bullets, one per item.\n"
        "- TASK question ('what tasks do I have', 'what's left to do', 'what's in progress', 'what "
        "have I finished', 'anything on my list'): answer ONLY from items in context whose body starts "
        "with 'Task:' — quote the title and status (e.g. 'You have two tasks left: Run backend sanity "
        "test (in progress) and Verify embedding index (todo).'). Do NOT use note prose for task "
        "questions, even if note content overlaps with a task title.\n"
        "- FOLLOW-UP turn: answer naturally building on what was just said.\n\n"
        "GUARDRAILS:\n"
        "- Vary your phrasing across turns. Never reuse the same opening pattern twice in a row.\n"
        "- The context can include both regular notes and task summaries. Treat task entries the same "
        "as notes — they are valid sources and you can cite them with their [N#] tag.\n"
        "- You may make small inferences (ordering events, inferring a role, naming a relationship) "
        "when supported by what you see. Never invent specifics that aren't there.\n"
        "- If a detail isn't in the context, omit it. Don't guess names, numbers, dates, or statuses.\n"
        "- If the context truly doesn't address the question, reply EXACTLY: 'Not found in your notes yet.'\n\n"
        "FORBIDDEN:\n"
        "- Padding a one-line factual answer with extra context the user didn't ask about. THIS IS THE "
        "MOST COMMON FAILURE — resist the urge to be 'helpful' by adding more.\n"
        "- Preamble: 'Based on the notes...', 'According to your notes...', 'It looks like...'.\n"
        "- Restating the question before answering.\n"
        "- The words 'span', 'context', 'block', 'document', 'snippet', or any UUID-style id.\n"
        "- Hedging filler ('I think', 'possibly', 'it seems') when the source is explicit.\n"
        "- Defaulting to bullets when prose would read better."
    )
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {req.question.strip()}\n\n"
        "Answer ONLY what was asked. If it's a direct lookup, one sentence. "
        "If it's a broad/synthesis question, fuse facts across items. Cite with [N#]."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in (req.history or [])[-8:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_prompt})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    cited = []
    seen = set()
    for m in re.finditer(r"\[(N\d+)\]", answer):
        sid = m.group(1)
        if sid in allowed_ids and sid not in seen:
            seen.add(sid)
            cited.append(sid)

    return {"answer": answer, "citations": cited, "model": model}


@app.post("/ml/notes/process", response_model=ProcessResp, dependencies=[Depends(require_key)])
def process_endpoint(req: ProcessReq):
    """Full end-to-end pipeline for a note's text.

    Runs chunk → embed → (optional) extract → (optional) insights, returns
    everything as JSON. The backend persists results to its own DB. Synchronous;
    expect ~10-60s depending on text length and LLM latency.
    """
    text = (req.text or "").strip()
    if not text:
        return {
            "spans": [], "embeddings": [], "objects": [], "links": [], "mentions": [],
            "contradictions": [], "stale_threads": [], "status": "empty",
        }

    # 1. Chunk
    spans = _chunk_text(text)

    # 2. Embed
    embeddings: List[List[float]] = []
    if spans:
        model = _get_embed_model()
        embeddings = model.encode(
            [s["text"] for s in spans], batch_size=32, show_progress_bar=False
        ).tolist()

    objects: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    mentions: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []
    stale: List[Dict[str, Any]] = []

    # 3. LLM extraction (optional)
    if req.run_extraction and (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        try:
            from ml.extraction import LLMExtractor
            result = LLMExtractor(verbose=False).extract(
                text, note_id=req.note_id or str(uuid4())
            )
            objects = [
                {"id": o.id, "type": o.type, "canonical_text": o.canonical_text, "confidence": o.confidence}
                for o in (result.objects or [])
            ]
            links = [
                {"source_id": l.source_id, "target_id": l.target_id, "type": l.type, "confidence": l.confidence}
                for l in (result.links or [])
            ]
            for m in (getattr(result, "mentions", None) or []):
                mentions.append({
                    "object_id": getattr(m, "object_id", None) or getattr(m, "id", ""),
                    "start_char": getattr(m, "start_char", None),
                    "end_char": getattr(m, "end_char", None),
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"extraction failed: {e}")

        # 4. Insights (contradictions only — stale_threads needs historical timestamps)
        if req.run_insights and objects:
            try:
                from ml.extraction import ExtractedObject, Link as MLLink
                from ml.graph import KnowledgeGraph
                from ml.intelligence import IntelligenceLayer
                kg = KnowledgeGraph()
                kg.add_objects([ExtractedObject(**o) for o in objects])
                kg.add_links([MLLink(**l) for l in links])
                contradictions = IntelligenceLayer(kg).detect_contradictions()
            except Exception as e:
                contradictions = [{"error": f"insight failure: {e}"}]

    return {
        "spans": spans,
        "embeddings": embeddings,
        "objects": objects,
        "links": links,
        "mentions": mentions,
        "contradictions": contradictions,
        "stale_threads": stale,
        "status": "ready",
    }


# ── File text extraction ────────────────────────────────────────────────────
@app.post("/ml/extract-text", dependencies=[Depends(require_key)])
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Accepts a binary file (PDF, DOCX, image, audio, video, plain text) and
    returns cleaned plain text. Implemented inline to stay decoupled from
    ml.db / ml.extraction_tasks (which are note_id- and SQLite-bound)."""
    data = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    name_lower = (file.filename or "").lower()
    mime = (file.content_type or "").lower()

    def _read_text(b: bytes) -> str:
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return b.decode("utf-8", errors="ignore")

    def _extract_pdf(path: Path) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                return "\n\n".join((p.extract_text() or "") for p in pdf.pages).strip()
        except Exception:
            pass
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(path))
            return "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
        except Exception:
            return ""

    def _extract_docx(path: Path) -> str:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    def _extract_image(path: Path) -> str:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(str(path))).strip()

    def _transcribe(path: Path) -> str:
        global _WHISPER
        if _WHISPER is None:
            with _WHISPER_LOCK:
                if _WHISPER is None:
                    import whisper
                    _WHISPER = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
        return (_WHISPER.transcribe(str(path)).get("text") or "").strip()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin")
    try:
        tmp.write(data)
        tmp.close()
        p = Path(tmp.name)

        if suffix == ".pdf" or mime == "application/pdf":
            text = _extract_pdf(p)
        elif suffix == ".docx" or "wordprocessingml" in mime:
            text = _extract_docx(p)
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"} or mime.startswith("image/"):
            text = _extract_image(p)
        elif suffix in {".mp3", ".wav", ".m4a", ".mp4"} or mime.startswith(("audio/", "video/")):
            text = _transcribe(p)
        else:
            text = _read_text(data)

        cleaned = re.sub(r"\r\n?|\r", "\n", text or "").strip()
        return {
            "cleaned_text": cleaned,
            "filename": file.filename,
            "mime_type": mime or "application/octet-stream",
            "char_count": len(cleaned),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extract-text failed: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ── HITL feedback (wraps ml/feedback.py) ────────────────────────────────────
class ReviewAction(BaseModel):
    action: Literal["accepted", "rejected", "corrected"]
    corrected_type: Optional[ObjectType] = None
    corrected_text: Optional[str] = None


@app.get("/ml/feedback/pending", dependencies=[Depends(require_key)])
def feedback_pending(note_id: Optional[str] = None):
    """List pending human-review items. `ml.extraction` auto-logs every
    extraction's objects as pending — call this to drive a review UI."""
    from ml.feedback import get_pending_reviews
    return {"pending": get_pending_reviews(note_id=note_id)}


@app.post("/ml/feedback/review/{review_id}", dependencies=[Depends(require_key)])
def feedback_review(review_id: int, body: ReviewAction):
    """Submit a human correction. The correction is used as a few-shot example
    by the next /ml/extract call."""
    from ml.feedback import submit_review
    try:
        submit_review(
            review_id=review_id,
            action=body.action,
            corrected_type=body.corrected_type,
            corrected_text=body.corrected_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "review_id": review_id, "action": body.action}


@app.get("/ml/feedback/stats", dependencies=[Depends(require_key)])
def feedback_stats():
    from ml.feedback import get_review_stats
    return get_review_stats()
