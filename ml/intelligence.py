from typing import List, Dict, Any, Optional
from datetime import datetime
from .graph import KnowledgeGraph
from .extraction import ExtractedObject, Link


class IntelligenceLayer:
    def __init__(self, graph: KnowledgeGraph):
        """
        Initialize the Intelligence Layer.

        Args:
            graph: The knowledge graph instance to analyze.
        """
        self.graph = graph

    # -----------------------------
    # Existing capability: contradictions
    # -----------------------------

    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Detects contradictions in the knowledge graph.

        Returns:
            A list of detected contradictions with severity and evidence.
        """
        explicit_contradictions = self.graph.find_contradictions()

        results = []
        for c in explicit_contradictions:
            results.append({
                "type": "Explicit",
                "source_text": c["source"]["canonical_text"],
                "target_text": c["target"]["canonical_text"],
                "severity": "High"
            })

        return results

    # -----------------------------
    # Existing capability: graph insights
    # -----------------------------

    def generate_insights(self) -> List[Dict[str, Any]]:
        """
        Generates insights based on graph structure.

        Returns:
            A list of insights.
        """
        insights = []

        # Insight 1: Stale Threads (Questions with no outgoing links)
        for node, data in self.graph.graph.nodes(data=True):
            if data.get("type") == "Question":
                if self.graph.graph.out_degree(node) == 0:
                    insights.append({
                        "type": "StaleThread",
                        "object_id": node,
                        "text": data.get("canonical_text"),
                        "message": "This question has not been addressed or linked to any answer."
                    })

        # Insight 2: High Centrality Ideas (Core themes)
        centrality = self.graph.custom_centrality()
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]

        for node_id, score in top_nodes:
            if score > 0:
                node_data = self.graph.graph.nodes[node_id]
                insights.append({
                    "type": "CoreConcept",
                    "object_id": node_id,
                    "text": node_data.get("canonical_text"),
                    "score": score,
                    "message": "This concept is central to the knowledge graph."
                })

        return insights

    # -----------------------------
    # NEW: task intelligence
    # -----------------------------

    def enrich_tasks(self, tasks: List[Dict[str, Any]], now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Adds computed fields to task records:
          - due_in_days
          - is_due_today
          - is_overdue
          - is_due_soon
          - urgency_score
          - attention_bucket
        """
        now_dt = now or datetime.utcnow()
        today = now_dt.date()

        enriched: List[Dict[str, Any]] = []
        for task in tasks:
            item = dict(task)

            status = (item.get("status") or "todo").lower()
            priority = (item.get("priority") or "medium").lower()
            due_date_str = item.get("due_date")

            due_in_days = None
            is_due_today = False
            is_overdue = False
            is_due_soon = False

            if due_date_str:
                try:
                    due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
                    due_in_days = (due_date - today).days
                    is_due_today = (due_in_days == 0)
                    is_overdue = (due_in_days < 0 and status not in {"done", "archived"})
                    is_due_soon = (0 <= due_in_days <= 3 and status not in {"done", "archived"})
                except ValueError:
                    # malformed dates are ignored for date-based signals
                    pass

            urgency_score = self._compute_urgency_score(
                priority=priority,
                status=status,
                due_in_days=due_in_days
            )

            item["due_in_days"] = due_in_days
            item["is_due_today"] = is_due_today
            item["is_overdue"] = is_overdue
            item["is_due_soon"] = is_due_soon
            item["urgency_score"] = urgency_score
            item["attention_bucket"] = self._attention_bucket(item)

            enriched.append(item)

        # High urgency first
        enriched.sort(
            key=lambda t: (-float(t.get("urgency_score", 0.0)), t.get("due_date") or "9999-12-31")
        )
        return enriched

    def summarize_tasks(
        self,
        tasks: List[Dict[str, Any]],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        now: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Produces dashboard-ready task summary.
        """
        scoped = self._scope_tasks(tasks, workspace_id=workspace_id, user_id=user_id, include_done=True)
        enriched = self.enrich_tasks(scoped, now=now)

        total = len(enriched)
        done = sum(1 for t in enriched if (t.get("status") or "").lower() == "done")
        in_progress = sum(1 for t in enriched if (t.get("status") or "").lower() == "in_progress")
        todo = sum(1 for t in enriched if (t.get("status") or "").lower() == "todo")
        archived = sum(1 for t in enriched if (t.get("status") or "").lower() == "archived")

        overdue = sum(1 for t in enriched if t.get("is_overdue"))
        due_today = sum(1 for t in enriched if t.get("is_due_today"))
        due_soon = sum(1 for t in enriched if t.get("is_due_soon"))

        by_priority = {
            "high": sum(1 for t in enriched if (t.get("priority") or "").lower() == "high"),
            "medium": sum(1 for t in enriched if (t.get("priority") or "").lower() == "medium"),
            "low": sum(1 for t in enriched if (t.get("priority") or "").lower() == "low"),
        }

        completion_rate = (done / total) if total > 0 else 0.0

        return {
            "total_tasks": total,
            "status_breakdown": {
                "todo": todo,
                "in_progress": in_progress,
                "done": done,
                "archived": archived
            },
            "priority_breakdown": by_priority,
            "overdue": overdue,
            "due_today": due_today,
            "due_soon": due_soon,
            "completion_rate": round(completion_rate, 4),
            "top_attention": enriched[:5]
        }

    def task_insights(
        self,
        tasks: List[Dict[str, Any]],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        now: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Human-readable insights for task panel.
        """
        scoped = self._scope_tasks(tasks, workspace_id=workspace_id, user_id=user_id, include_done=True)
        enriched = self.enrich_tasks(scoped, now=now)

        insights: List[Dict[str, Any]] = []

        overdue_tasks = [t for t in enriched if t.get("is_overdue")]
        if overdue_tasks:
            insights.append({
                "type": "OverdueTasks",
                "severity": "High",
                "count": len(overdue_tasks),
                "message": f"{len(overdue_tasks)} task(s) are overdue."
            })

        today_tasks = [t for t in enriched if t.get("is_due_today")]
        if today_tasks:
            insights.append({
                "type": "DueToday",
                "severity": "Medium",
                "count": len(today_tasks),
                "message": f"{len(today_tasks)} task(s) are due today."
            })

        high_open = [
            t for t in enriched
            if (t.get("priority") or "").lower() == "high"
            and (t.get("status") or "").lower() not in {"done", "archived"}
        ]
        if high_open:
            insights.append({
                "type": "HighPriorityOpen",
                "severity": "Medium",
                "count": len(high_open),
                "message": f"{len(high_open)} high-priority task(s) are still open."
            })

        # Completion trend snapshot (simple point-in-time)
        total = len(enriched)
        done = sum(1 for t in enriched if (t.get("status") or "").lower() == "done")
        if total > 0:
            rate = done / total
            insights.append({
                "type": "CompletionSnapshot",
                "severity": "Info",
                "value": round(rate, 4),
                "message": f"Current completion rate is {round(rate * 100, 1)}%."
            })

        return insights

    # -----------------------------
    # Optional: read tasks directly from graph nodes
    # -----------------------------

    def collect_tasks_from_graph(
        self,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Pull task records directly from graph node attributes.
        Useful when no external task store is available.
        """
        tasks: List[Dict[str, Any]] = []
        for node_id, data in self.graph.graph.nodes(data=True):
            if data.get("type") != "Task":
                continue

            if workspace_id is not None and data.get("workspace_id") != workspace_id:
                continue
            if user_id is not None and data.get("user_id") != user_id:
                continue

            tasks.append({
                "id": node_id,
                "title": data.get("title") or data.get("canonical_text"),
                "canonical_text": data.get("canonical_text"),
                "status": data.get("status", "todo"),
                "priority": data.get("priority", "medium"),
                "due_date": data.get("due_date"),
                "due_time": data.get("due_time"),
                "note_id": data.get("note_id"),
                "user_id": data.get("user_id"),
                "workspace_id": data.get("workspace_id"),
                "confidence": data.get("confidence"),
                "source_text": data.get("source_text"),
            })

        return tasks

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _scope_tasks(
        self,
        tasks: List[Dict[str, Any]],
        workspace_id: Optional[str],
        user_id: Optional[str],
        include_done: bool
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in tasks:
            if workspace_id is not None and t.get("workspace_id") != workspace_id:
                continue
            if user_id is not None and t.get("user_id") != user_id:
                continue
            if not include_done and (t.get("status") or "").lower() == "done":
                continue
            out.append(t)
        return out

    def _compute_urgency_score(self, priority: str, status: str, due_in_days: Optional[int]) -> float:
        base = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(priority, 0.5)

        if status in {"done", "archived"}:
            return 0.0

        score = base
        if due_in_days is None:
            score += 0.05
        elif due_in_days < 0:
            lateness = min(abs(due_in_days), 14)
            score += 0.35 + (lateness / 14.0) * 0.30
        elif due_in_days == 0:
            score += 0.30
        elif due_in_days <= 3:
            score += 0.20
        elif due_in_days <= 7:
            score += 0.10
        else:
            score += 0.02

        if status == "in_progress":
            score += 0.05

        return round(min(score, 1.0), 4)

    def _attention_bucket(self, task: Dict[str, Any]) -> str:
        status = (task.get("status") or "").lower()
        if status in {"done", "archived"}:
            return "closed"
        if task.get("is_overdue"):
            return "overdue"
        if task.get("is_due_today"):
            return "today"
        if task.get("is_due_soon"):
            return "soon"
        return "planned"
