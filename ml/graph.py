import networkx as nx
from typing import List, Dict, Optional
import json

from .extraction import ExtractedObject, Link

class KnowledgeGraph:
    def __init__(self):
        """Initialize an empty Knowledge Graph using NetworkX."""
        self.graph = nx.DiGraph()

    def add_objects(self, objects: List[ExtractedObject]):
        """Adds extracted objects as nodes to the graph."""
        for obj in objects:
            self.graph.add_node(
                obj.id,
                type=obj.type,
                canonical_text=obj.canonical_text,
                confidence=obj.confidence
            )

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

    def get_subgraph(self, node_id: str, depth: int = 1) -> Dict:
        """
        Retrieves a subgraph around a specific node.
        
        Args:
            node_id: The extracted object ID to center the subgraph on.
            depth: How many hops to traverse.
            
        Returns:
            A dictionary representation of the subgraph (nodes and links).
        """
        if node_id not in self.graph:
            return {"nodes": [], "links": []}

        # BFS to find nodes within depth
        subgraph_nodes = set(nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth).keys())
        # Also include predecessors for context if needed, but for now just downstream
        
        # Create a subgraph view
        sub_G = self.graph.subgraph(subgraph_nodes)
        
        return nx.node_link_data(sub_G)

    def find_contradictions(self) -> List[Dict]:
        """
        Simple heuristic to find potential contradictions based on edge types.
        """
        contradictions = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('type') == 'Contradicts':
                contradictions.append({
                    "source": self.graph.nodes[u],
                    "target": self.graph.nodes[v],
                    "edge": data
                })
        return contradictions

    def custom_centrality(self):
        """Calculates centrality to find core ideas."""
        return nx.degree_centrality(self.graph)
