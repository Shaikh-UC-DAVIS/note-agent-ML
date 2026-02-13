import networkx as nx
from typing import List, Dict, Optional, Any
import json

from .extraction import ExtractedObject, Link


class KnowledgeGraph:
    def __init__(self):
        """Initialize an empty Knowledge Graph using NetworkX."""
        self.graph = nx.DiGraph()

    def add_objects(self, objects: List[ExtractedObject]):
        """
        Adds extracted objects as nodes to the graph.

        Important:
        - Preserves optional object.attributes (e.g., due_date, note_id, user_id, workspace_id).
        - Flattens attributes into node properties for easier querying.
        """
        for obj in objects:
            node_data = {
                "type": obj.type,
                "canonical_text": obj.canonical_text,
                "confidence": obj.confidence,
            }

            # Preserve the original attributes dict if present
            if getattr(obj, "attributes", None):
                node_data["attributes"] = obj.attributes

                # Also flatten for convenience in graph queries
                if isinstance(obj.attributes, dict):
                    for k, v in obj.attributes.items():
                        # avoid overwriting reserved keys
                        if k not in node_data:
                            node_data[k] = v

            self.graph.add_node(obj.id, **node_data)

    def add_links(self, links: List[Link]):
        """Adds links as directed edges to the graph."""
        for link in links:
            if self.graph.has_node(link.source_id) and self.graph.has_node(link.target_id):
                self.graph.add_edge(
                    link.source_id,
                    link.target_id,
                    type=link.type,
                    confidence=link.confidence
                )
            else:
                # In a real system you might log a warning or handle missing nodes
                pass

    def get_subgraph(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Retrieves a subgraph around a specific node.

        Args:
            node_id: The extracted object ID to center the subgraph on.
            depth: How many hops to traverse.

        Returns:
            Node-link dictionary representation of the subgraph.
        """
        if node_id not in self.graph:
            return {"nodes": [], "links": []}

        # downstream nodes within depth
        subgraph_nodes = set(
            nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth).keys()
        )

        # Optional context: include direct predecessors to provide upstream explanation
        for pred in self.graph.predecessors(node_id):
            subgraph_nodes.add(pred)

        sub_G = self.graph.subgraph(subgraph_nodes)
        return nx.node_link_data(sub_G)

    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Simple heuristic to find potential contradictions based on edge types."""
        contradictions = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == "Contradicts":
                contradictions.append({
                    "source_id": u,
                    "target_id": v,
                    "source": dict(self.graph.nodes[u]),
                    "target": dict(self.graph.nodes[v]),
                    "edge": dict(data)
                })
        return contradictions

    def custom_centrality(self) -> Dict[str, float]:
        """Calculates degree centrality to find core ideas."""
        return nx.degree_centrality(self.graph)

    # -----------------------------
    # New task-focused helpers
    # -----------------------------

    def get_task_nodes(
        self,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        due_date: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns task nodes with optional filters.

        Args:
            workspace_id: filter tasks by workspace_id
            user_id: filter tasks by user_id (creator/owner)
            due_date: filter exact due_date (YYYY-MM-DD)
            status: filter status (todo/in_progress/done/etc.)
        """
        tasks: List[Dict[str, Any]] = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "Task":
                continue

            if workspace_id is not None and data.get("workspace_id") != workspace_id:
                continue
            if user_id is not None and data.get("user_id") != user_id:
                continue
            if due_date is not None and data.get("due_date") != due_date:
                continue
            if status is not None and data.get("status") != status:
                continue

            tasks.append({
                "id": node_id,
                **dict(data)
            })

        return tasks

    def get_tasks_for_date_range(
        self,
        start_date: str,
        end_date: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_without_due_date: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Returns tasks with due_date in [start_date, end_date] lexicographically.
        Works because dates are expected in ISO format YYYY-MM-DD.
        """
        tasks: List[Dict[str, Any]] = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "Task":
                continue

            if workspace_id is not None and data.get("workspace_id") != workspace_id:
                continue
            if user_id is not None and data.get("user_id") != user_id:
                continue

            d = data.get("due_date")
            if d is None:
                if include_without_due_date:
                    tasks.append({"id": node_id, **dict(data)})
                continue

            if start_date <= d <= end_date:
                tasks.append({"id": node_id, **dict(data)})

        # Sort by due_date then due_time (None-safe)
        tasks.sort(key=lambda x: (x.get("due_date") or "9999-12-31", x.get("due_time") or "23:59"))
        return tasks

    def attach_note_to_tasks(self, note_object_id: str, task_ids: List[str], confidence: float = 0.9):
        """
        Utility to explicitly connect a note node to its extracted tasks.
        Edge type chosen from allowed link enum: 'Refines' (note refined into actionable tasks).
        """
        if not self.graph.has_node(note_object_id):
            return

        for tid in task_ids:
            if self.graph.has_node(tid):
                self.graph.add_edge(
                    note_object_id,
                    tid,
                    type="Refines",
                    confidence=confidence
                )

    def to_json(self, indent: int = 2) -> str:
        """Serialize full graph to JSON node-link format."""
        return json.dumps(nx.node_link_data(self.graph), indent=indent)

    def from_json(self, payload: str):
        """Load graph from JSON node-link format."""
        data = json.loads(payload)
        self.graph = nx.node_link_graph(data, directed=True)


## Changes Made:

'''

Updated add_objects(...) to preserve and store obj.attributes on nodes.

Flattened attributes into node properties for easier filtering/querying.

Added task utilities:

get_task_nodes(...)

get_tasks_for_date_range(...)

attach_note_to_tasks(...)

Kept existing contradiction/centrality/subgraph functionality.

Why

Previously, graph nodes only stored type, canonical_text, confidence; task due dates and ownership were lost.

Calendar queries need filters like:

by workspace_id

by user_id

by due_date range

This enables direct “tasks due this week/month” retrieval from graph state.

'''