from typing import List, Dict, Any
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

    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Detects contradictions in the knowledge graph.
        
        Returns:
            A list of detected contradictions with severity and evidence.
        """
        # 1. Structural contradictions (explicit edges)
        explicit_contradictions = self.graph.find_contradictions()
        
        # 2. Semantic contradictions (mocked logic)
        # Real implementation would compare embeddings of all Claims
        # and use an LLM to verify if high-similarity claims are contradictory.
        
        results = []
        for c in explicit_contradictions:
            results.append({
                "type": "Explicit",
                "source_text": c['source']['canonical_text'],
                "target_text": c['target']['canonical_text'],
                "severity": "High"
            })
            
        return results

    def generate_insights(self) -> List[Dict[str, Any]]:
        """
        Generates insights based on graph structure.
        
        Returns:
            A list of insights.
        """
        insights = []
        
        # Insight 1: Stale Threads (Questions with no answers/links)
        for node, data in self.graph.graph.nodes(data=True):
            if data.get('type') == 'Question':
                # Check if it has any outgoing edges (answers/refinements)
                if self.graph.graph.out_degree(node) == 0:
                    insights.append({
                        "type": "StaleThread",
                        "object_id": node,
                        "text": data.get('canonical_text'),
                        "message": "This question has not been addressed or linked to any answer."
                    })

        # Insight 2: High Centrality Ideas (Core themes)
        centrality = self.graph.custom_centrality()
        # Sort by centrality
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for node_id, score in top_nodes:
            if score > 0:
                node_data = self.graph.graph.nodes[node_id]
                insights.append({
                    "type": "CoreConcept",
                    "object_id": node_id,
                    "text": node_data.get('canonical_text'),
                    "score": score,
                    "message": "This concept is central to the knowledge graph."
                })
                
        return insights
