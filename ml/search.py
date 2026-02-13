from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    source: str  # 'vector', 'keyword', 'graph', 'hybrid'


@dataclass
class TaskSearchResult:
    task_id: str
    title: str
    score: float
    source: str  # 'nl_filter', 'structured', 'text'
    task: Dict[str, Any]


class HybridSearchEngine:
    def __init__(self, embedding_generator=None, graph=None):
        """
        Initialize the Hybrid Search Engine.

        Args:
            embedding_generator: Instance of EmbeddingGenerator (from ingestion).
            graph: Instance of KnowledgeGraph (from graph).
        """
        self.embedding_generator = embedding_generator
        self.graph = graph
        self.chunks: List[Dict[str, Any]] = []  # chunk index

    # -----------------------------
    # Chunk indexing + retrieval
    # -----------------------------

    def index_chunk(self, chunk_id: str, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Adds a chunk to the search index."""
        self.chunks.append({
            "id": chunk_id,
            "text": text,
            "vector": vector,
            "metadata": metadata or {}
        })

    def _safe_cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Numerically safe cosine similarity."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _vector_search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """Performs cosine similarity search."""
        if not self.chunks:
            return []

        results = []
        q_vec = np.array(query_vector)

        for chunk in self.chunks:
            c_vec = np.array(chunk["vector"])
            similarity = self._safe_cosine(q_vec, c_vec)

            results.append(SearchResult(
                chunk_id=chunk["id"],
                text=chunk["text"],
                score=similarity,
                source="vector"
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _keyword_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Performs simple keyword search."""
        results = []
        query_terms = set(query.lower().split())

        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            score = 0
            for term in query_terms:
                if term in text_lower:
                    score += 1

            if score > 0:
                results.append(SearchResult(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    score=float(score * 10),  # boost keyword score in simple fusion
                    source="keyword"
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _graph_context_boost(self, results: List[SearchResult], depth: int = 1, boost: float = 0.15) -> List[SearchResult]:
        """
        Optional rerank boost using graph presence.
        Expects chunk IDs may correspond to graph node IDs in some setups.
        """
        if not self.graph:
            return results

        boosted: List[SearchResult] = []
        for r in results:
            s = r.score
            try:
                sub = self.graph.get_subgraph(r.chunk_id, depth=depth)
                # if graph context exists, mild score boost
                if sub.get("nodes"):
                    s = s + boost
                    src = "graph" if r.source != "hybrid" else "hybrid"
                else:
                    src = r.source
            except Exception:
                src = r.source

            boosted.append(SearchResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=s,
                source=src
            ))

        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    def search(self, query: str, top_k: int = 5, use_graph_boost: bool = True) -> List[SearchResult]:
        """
        Performs hybrid chunk search:
          1) vector retrieval (if embeddings enabled)
          2) keyword retrieval
          3) score fusion + dedupe
          4) optional graph-context boost
        """
        # 1) Vector search
        if self.embedding_generator:
            query_vector = self.embedding_generator.model.encode(query)
            vector_results = self._vector_search(query_vector, top_k=top_k)
        else:
            vector_results = []

        # 2) Keyword search
        keyword_results = self._keyword_search(query, top_k=top_k)

        # 3) Fusion
        combined = {r.chunk_id: r for r in vector_results}
        for res in keyword_results:
            if res.chunk_id in combined:
                combined_res = combined[res.chunk_id]
                combined_res.score += res.score
                combined_res.source = "hybrid"
            else:
                combined[res.chunk_id] = res

        final_results = list(combined.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        final_results = final_results[:top_k]

        # 4) Graph boost (optional)
        if use_graph_boost:
            final_results = self._graph_context_boost(final_results)

        return final_results

    # -----------------------------
    # NEW: Task-focused search
    # -----------------------------

    def search_tasks_nl(
        self,
        tasks: List[Dict[str, Any]],
        query: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_done: bool = False
    ) -> List[TaskSearchResult]:
        """
        Natural-language-ish task filtering.
        Supports intents like:
          - "overdue"
          - "today"
          - "tomorrow"
          - "this week"
          - "high priority"
          - "from note note-123"
          - fallback text search
        """
        q = (query or "").strip().lower()
        scoped = self._scope_tasks(tasks, workspace_id, user_id, include_done)

        # enrich first for due/overdue logic
        enriched = [self._enrich_task(t) for t in scoped]

        if "overdue" in q:
            filtered = [t for t in enriched if t["is_overdue"]]
            return self._to_task_results(filtered, source="nl_filter")

        if "today" in q:
            filtered = [t for t in enriched if t["is_due_today"]]
            return self._to_task_results(filtered, source="nl_filter")

        if "tomorrow" in q:
            target = (datetime.utcnow().date() + timedelta(days=1)).isoformat()
            filtered = [t for t in enriched if t.get("due_date") == target]
            return self._to_task_results(filtered, source="nl_filter")

        if "this week" in q or "week" in q:
            start, end = self._current_week_range()
            filtered = [t for t in enriched if t.get("due_date") and start <= t["due_date"] <= end]
            return self._to_task_results(filtered, source="nl_filter")

        if "high priority" in q or "priority high" in q:
            filtered = [t for t in enriched if (t.get("priority") or "").lower() == "high"]
            return self._to_task_results(filtered, source="nl_filter")

        note_id = self._extract_note_id(q)
        if note_id:
            filtered = [t for t in enriched if (t.get("note_id") or "").lower() == note_id.lower()]
            return self._to_task_results(filtered, source="nl_filter")

        # fallback lexical search over task text
        filtered = self._task_text_search(enriched, q)
        return self._to_task_results(filtered, source="text")

    def search_tasks_structured(
        self,
        tasks: List[Dict[str, Any]],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        note_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_done: bool = True
    ) -> List[TaskSearchResult]:
        """
        Structured task filtering for backend controllers.
        """
        out = self._scope_tasks(tasks, workspace_id, user_id, include_done)

        if status:
            out = [t for t in out if (t.get("status") or "").lower() == status.lower()]
        if priority:
            out = [t for t in out if (t.get("priority") or "").lower() == priority.lower()]
        if note_id:
            out = [t for t in out if (t.get("note_id") or "") == note_id]
        if start_date and end_date:
            out = [t for t in out if t.get("due_date") and start_date <= t["due_date"] <= end_date]

        enriched = [self._enrich_task(t) for t in out]
        return self._to_task_results(enriched, source="structured")

    def summarize_tasks(
        self,
        tasks: List[Dict[str, Any]],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dashboard/task-panel summary.
        """
        scoped = self._scope_tasks(tasks, workspace_id, user_id, include_done=True)
        enriched = [self._enrich_task(t) for t in scoped]

        total = len(enriched)
        done = sum(1 for t in enriched if (t.get("status") or "").lower() == "done")
        in_progress = sum(1 for t in enriched if (t.get("status") or "").lower() == "in_progress")
        todo = sum(1 for t in enriched if (t.get("status") or "").lower() == "todo")
        archived = sum(1 for t in enriched if (t.get("status") or "").lower() == "archived")

        overdue = sum(1 for t in enriched if t["is_overdue"])
        due_today = sum(1 for t in enriched if t["is_due_today"])
        due_soon = sum(1 for t in enriched if t["is_due_soon"])

        completion_rate = (done / total) if total else 0.0

        top_attention = sorted(
            enriched, key=lambda x: x.get("urgency_score", 0.0), reverse=True
        )[:5]

        return {
            "total_tasks": total,
            "status_breakdown": {
                "todo": todo,
                "in_progress": in_progress,
                "done": done,
                "archived": archived,
            },
            "overdue": overdue,
            "due_today": due_today,
            "due_soon": due_soon,
            "completion_rate": round(completion_rate, 4),
            "top_attention": top_attention,
        }

    # -----------------------------
    # Internal helpers (tasks)
    # -----------------------------

    def _scope_tasks(
        self,
        tasks: List[Dict[str, Any]],
        workspace_id: Optional[str],
        user_id: Optional[str],
        include_done: bool
    ) -> List[Dict[str, Any]]:
        out = []
        for t in tasks:
            if workspace_id is not None and t.get("workspace_id") != workspace_id:
                continue
            if user_id is not None and t.get("user_id") != user_id:
                continue
            if not include_done and (t.get("status") or "").lower() == "done":
                continue
            out.append(t)
        return out

    def _enrich_task(self, t: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(t)
        due_date = item.get("due_date")
        status = (item.get("status") or "todo").lower()
        priority = (item.get("priority") or "medium").lower()

        due_in_days: Optional[int] = None
        is_due_today = False
        is_overdue = False
        is_due_soon = False

        if due_date:
            try:
                d = datetime.strptime(due_date, "%Y-%m-%d").date()
                due_in_days = (d - datetime.utcnow().date()).days
                is_due_today = (due_in_days == 0)
                is_overdue = (due_in_days < 0 and status not in {"done", "archived"})
                is_due_soon = (0 <= due_in_days <= 3 and status not in {"done", "archived"})
            except ValueError:
                pass

        base = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(priority, 0.5)
        urgency = base
        if status in {"done", "archived"}:
            urgency = 0.0
        else:
            if due_in_days is None:
                urgency += 0.05
            elif due_in_days < 0:
                urgency += 0.45
            elif due_in_days == 0:
                urgency += 0.30
            elif due_in_days <= 3:
                urgency += 0.20
            elif due_in_days <= 7:
                urgency += 0.10
            if status == "in_progress":
                urgency += 0.05

        item["due_in_days"] = due_in_days
        item["is_due_today"] = is_due_today
        item["is_overdue"] = is_overdue
        item["is_due_soon"] = is_due_soon
        item["urgency_score"] = round(min(urgency, 1.0), 4)
        return item

    def _task_text_search(self, tasks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if not query:
            return tasks

        terms = [t for t in query.split() if t]
        scored: List[Tuple[int, Dict[str, Any]]] = []

        for t in tasks:
            hay = " ".join([
                str(t.get("title") or ""),
                str(t.get("canonical_text") or ""),
                str(t.get("source_text") or "")
            ]).lower()
            score = 0
            for term in terms:
                if term in hay:
                    score += 1
            if score > 0:
                scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored]

    def _extract_note_id(self, q: str) -> Optional[str]:
        if "note:" in q:
            return q.split("note:")[-1].split()[0].strip()
        if "from note " in q:
            return q.split("from note ")[-1].split()[0].strip()
        return None

    def _current_week_range(self) -> Tuple[str, str]:
        today = datetime.utcnow().date()
        start = today - timedelta(days=today.weekday())   # Monday
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()

    def _to_task_results(self, tasks: List[Dict[str, Any]], source: str) -> List[TaskSearchResult]:
        # sort by urgency first, then due date
        tasks_sorted = sorted(
            tasks,
            key=lambda t: (-float(t.get("urgency_score", 0.0)), t.get("due_date") or "9999-12-31")
        )

        out: List[TaskSearchResult] = []
        for t in tasks_sorted:
            title = str(t.get("title") or t.get("canonical_text") or "Untitled Task")
            out.append(TaskSearchResult(
                task_id=str(t.get("id", "")),
                title=title,
                score=float(t.get("urgency_score", 0.0)),
                source=source,
                task=t
            ))
        return out
