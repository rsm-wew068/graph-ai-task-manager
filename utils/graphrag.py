#!/usr/bin/env python3
"""
Enhanced GraphRAG implementation with semantic capabilities.
This replaces the naive graph querying in tools.py with true semantic reasoning.
"""
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np


class GraphRAG:
    """
    True GraphRAG implementation with semantic search and multi-path reasoning.
    
    This class provides semantic understanding of graph structures,
    multi-path exploration, and evidence aggregation for better retrieval.
    """
    
    def __init__(self, graph_path: str = "topic_graph.gpickle"):
        self.graph_path = graph_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph: Optional[nx.DiGraph] = None
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self._embeddings_computed = False
        
    def load_graph_with_embeddings(self) -> bool:
        """
        Load graph and compute semantic embeddings for all nodes.
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        if not os.path.exists(self.graph_path):
            return False
            
        try:
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)
            
            # Compute embeddings for all nodes if not already done
            if not self._embeddings_computed:
                self._compute_node_embeddings()
                
            return True
        except Exception:
            return False
    
    def _compute_node_embeddings(self):
        """Compute semantic embeddings for all graph nodes."""
        if not self.graph:
            return
            
        for node, attrs in self.graph.nodes(data=True):
            text = self._node_to_semantic_text(node, attrs)
            embedding = self.embedder.encode(text)
            self.node_embeddings[node] = embedding
            
        self._embeddings_computed = True
    
    def _node_to_semantic_text(self, node: str, attrs: Dict) -> str:
        """
        Convert node and its attributes to semantic text representation.
        
        Args:
            node: Node identifier
            attrs: Node attributes
            
        Returns:
            str: Semantic text representation
        """
        label = attrs.get("label", "")
        name = attrs.get("name", str(node))
        
        # Create rich semantic representation
        if label == "Task":
            return f"Task: {name}"
        elif label == "Person":
            return f"Person: {name}"
        elif label == "Topic":
            return f"Project Topic: {name}"
        elif label == "Organization":
            return f"Organization: {name}"
        elif label == "Role":
            return f"Job Role: {name}"
        elif label == "Department":
            return f"Department: {name}"
        elif label == "Date":
            return f"Date: {name}"
        else:
            return f"{label}: {name}"
    
    def semantic_node_search(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find nodes most semantically similar to the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (node, similarity_score) tuples
        """
        if not self.node_embeddings:
            return []
            
        query_embedding = self.embedder.encode(query)
        
        similarities = []
        for node, embedding in self.node_embeddings.items():
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            if sim >= min_similarity:
                similarities.append((node, sim))
        
        # Return top-k most similar nodes
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def multi_path_reasoning(
        self, 
        start_nodes: List[str], 
        max_hops: int = 4
    ) -> Dict[str, Dict]:
        """
        Explore multiple reasoning paths from start nodes.
        
        Args:
            start_nodes: List of starting nodes for exploration
            max_hops: Maximum number of hops to explore
            
        Returns:
            Dict mapping start_nodes to their reasoning paths
        """
        reasoning_paths = {}
        
        for start_node in start_nodes:
            if start_node not in self.graph:
                continue
                
            paths = {
                'shortest_paths': self._find_shortest_paths(start_node, max_hops),
                'centrality_paths': self._find_centrality_based_paths(start_node),
                'community_paths': self._find_community_based_paths(start_node),
                'typed_paths': self._find_typed_relationship_paths(start_node)
            }
            reasoning_paths[start_node] = paths
        
        return reasoning_paths
    
    def _find_shortest_paths(self, start_node: str, max_hops: int) -> List[List[str]]:
        """Find shortest paths to all reachable nodes within max_hops."""
        paths = []
        try:
            # Get all nodes within max_hops
            subgraph = nx.ego_graph(self.graph, start_node, radius=max_hops)
            
            # Find paths to all reachable nodes
            for target in subgraph.nodes():
                if target != start_node:
                    try:
                        path = nx.shortest_path(self.graph, start_node, target)
                        if len(path) <= max_hops + 1:
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        except Exception:
            pass
        return paths
    
    def _find_centrality_based_paths(self, start_node: str) -> List[List[str]]:
        """Find paths to high-centrality (important) nodes."""
        try:
            # Calculate different centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # Combine centrality scores
            combined_centrality = {}
            for node in self.graph.nodes():
                score = (degree_centrality.get(node, 0) + 
                        betweenness_centrality.get(node, 0)) / 2
                combined_centrality[node] = score
            
            # Get top central nodes
            high_centrality_nodes = sorted(
                combined_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            paths = []
            for node, _ in high_centrality_nodes:
                if node != start_node:
                    try:
                        path = nx.shortest_path(self.graph, start_node, node)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            return paths
        except Exception:
            return []
    
    def _find_community_based_paths(self, start_node: str) -> List[List[str]]:
        """Find paths within the same community/cluster."""
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()
            communities = list(nx.connected_components(undirected))
            
            # Find community containing start_node
            start_community = None
            for community in communities:
                if start_node in community:
                    start_community = community
                    break
            
            if start_community and len(start_community) > 1:
                subgraph = self.graph.subgraph(start_community)
                # Get DFS paths within community
                edges = list(nx.dfs_edges(subgraph, start_node, depth_limit=3))
                paths = []
                current_path = [start_node]
                for edge in edges:
                    if edge[0] == current_path[-1]:
                        current_path.append(edge[1])
                    else:
                        paths.append(current_path[:])
                        current_path = [edge[0], edge[1]]
                if current_path:
                    paths.append(current_path)
                return paths
        except Exception:
            pass
        return []
    
    def _find_typed_relationship_paths(self, start_node: str) -> Dict[str, List[List[str]]]:
        """Find paths based on specific relationship types."""
        typed_paths = {
            'ownership': [],
            'collaboration': [],
            'temporal': [],
            'organizational': []
        }
        
        try:
            # Explore different relationship types
            for neighbor in self.graph.neighbors(start_node):
                edge_data = self.graph[start_node][neighbor]
                label = edge_data.get('label', '')
                
                if label in ['RESPONSIBLE_TO', 'HAS_TASK']:
                    typed_paths['ownership'].append([start_node, neighbor])
                elif label in ['COLLABORATED_BY']:
                    typed_paths['collaboration'].append([start_node, neighbor])
                elif label in ['START_ON', 'DUE_ON']:
                    typed_paths['temporal'].append([start_node, neighbor])
                elif label in ['BELONGS_TO', 'IS_IN']:
                    typed_paths['organizational'].append([start_node, neighbor])
        except Exception:
            pass
            
        return typed_paths
    
    def query_with_semantic_reasoning(self, query: str) -> Dict:
        """
        Main GraphRAG query function with semantic reasoning.
        
        Args:
            query: Natural language query
            
        Returns:
            Dict containing structured response with evidence and reasoning
        """
        if not self.load_graph_with_embeddings():
            return {
                'query': query,
                'error': 'No topic graph found. Please process some emails first.',
                'evidence': {},
                'confidence_score': 0.0
            }
        
        # Step 1: Semantic search for relevant starting nodes
        relevant_nodes = self.semantic_node_search(query, top_k=5)
        
        if not relevant_nodes:
            return {
                'query': query,
                'error': 'No semantically relevant nodes found.',
                'evidence': {},
                'confidence_score': 0.0
            }
        
        # Step 2: Multi-path reasoning from relevant nodes
        start_nodes = [node for node, _ in relevant_nodes[:3]]  # Top 3 most relevant
        reasoning_paths = self.multi_path_reasoning(start_nodes)
        
        # Step 3: Aggregate and rank evidence
        evidence = self._aggregate_evidence(reasoning_paths, query, relevant_nodes)
        
        # Step 4: Calculate confidence score
        confidence = self._calculate_confidence(evidence, relevant_nodes)
        
        # Step 5: Format structured response
        return {
            'query': query,
            'evidence': evidence,
            'confidence_score': confidence,
            'reasoning_explanation': f"Found {len(relevant_nodes)} relevant nodes, explored {len(reasoning_paths)} reasoning paths",
            'relevant_nodes': [(node, score) for node, score in relevant_nodes]
        }
    
    def _aggregate_evidence(
        self, 
        reasoning_paths: Dict, 
        query: str, 
        relevant_nodes: List[Tuple[str, float]]
    ) -> Dict:
        """Aggregate evidence from multiple reasoning paths."""
        evidence = {
            'tasks': [],
            'people': [],
            'deadlines': [],
            'organizations': [],
            'relationships': []
        }
        
        # Collect all nodes found in reasoning paths
        all_path_nodes = set()
        for start_node, paths in reasoning_paths.items():
            for path_type, path_list in paths.items():
                if isinstance(path_list, dict):  # Handle typed paths
                    for relation_type, relation_paths in path_list.items():
                        for path in relation_paths:
                            all_path_nodes.update(path)
                else:  # Handle regular paths
                    for path in path_list:
                        all_path_nodes.update(path)
        
        # Categorize evidence by node type
        for node in all_path_nodes:
            if node in self.graph:
                attrs = self.graph.nodes[node]
                label = attrs.get('label', '')
                name = attrs.get('name', str(node))
                
                if label == 'Task':
                    evidence['tasks'].append({
                        'name': name,
                        'node': node,
                        'details': self._get_task_details(node)
                    })
                elif label == 'Person':
                    evidence['people'].append({
                        'name': name,
                        'node': node,
                        'details': self._get_person_details(node)
                    })
                elif label == 'Date':
                    evidence['deadlines'].append({
                        'date': name,
                        'node': node,
                        'context': self._get_date_context(node)
                    })
                elif label == 'Organization':
                    evidence['organizations'].append({
                        'name': name,
                        'node': node
                    })
        
        return evidence
    
    def _get_task_details(self, task_node: str) -> Dict:
        """Get detailed information about a task."""
        details = {'start_date': None, 'due_date': None, 'owner': None, 'summary': None}
        
        try:
            for neighbor in self.graph.neighbors(task_node):
                edge_label = self.graph[task_node][neighbor].get('label', '')
                neighbor_attrs = self.graph.nodes[neighbor]
                
                if edge_label == 'START_ON':
                    details['start_date'] = neighbor_attrs.get('name', neighbor)
                elif edge_label == 'DUE_ON':
                    details['due_date'] = neighbor_attrs.get('name', neighbor)
                elif edge_label == 'RESPONSIBLE_TO':
                    details['owner'] = neighbor_attrs.get('name', neighbor)
                elif edge_label == 'BASED_ON':
                    details['summary'] = neighbor_attrs.get('name', neighbor)
        except Exception:
            pass
            
        return details
    
    def _get_person_details(self, person_node: str) -> Dict:
        """Get detailed information about a person."""
        details = {'role': None, 'department': None, 'organization': None}
        
        try:
            for neighbor in self.graph.neighbors(person_node):
                edge_label = self.graph[person_node][neighbor].get('label', '')
                neighbor_attrs = self.graph.nodes[neighbor]
                
                if edge_label == 'HAS_ROLE':
                    details['role'] = neighbor_attrs.get('name', neighbor)
        except Exception:
            pass
            
        return details
    
    def _get_date_context(self, date_node: str) -> List[str]:
        """Get context for what happens on a specific date."""
        context = []
        
        try:
            # Find tasks that start or are due on this date
            for predecessor in self.graph.predecessors(date_node):
                edge_label = self.graph[predecessor][date_node].get('label', '')
                task_name = self.graph.nodes[predecessor].get('name', predecessor)
                
                if edge_label == 'START_ON':
                    context.append(f"Task '{task_name}' starts")
                elif edge_label == 'DUE_ON':
                    context.append(f"Task '{task_name}' due")
        except Exception:
            pass
            
        return context
    
    def _calculate_confidence(
        self, 
        evidence: Dict, 
        relevant_nodes: List[Tuple[str, float]]
    ) -> float:
        """Calculate confidence score based on evidence quality."""
        if not relevant_nodes:
            return 0.0
        
        # Base confidence from semantic similarity
        max_similarity = max(score for _, score in relevant_nodes)
        
        # Boost confidence based on evidence richness
        evidence_count = (
            len(evidence.get('tasks', [])) +
            len(evidence.get('people', [])) +
            len(evidence.get('deadlines', [])) +
            len(evidence.get('organizations', []))
        )
        
        # Normalize to 0-1 range
        confidence = min(1.0, max_similarity + (evidence_count * 0.1))
        
        return round(confidence, 3)


def format_graphrag_response(graphrag_result: Dict) -> str:
    """
    Format GraphRAG response into human-readable text.
    
    Args:
        graphrag_result: Result from GraphRAG query
        
    Returns:
        str: Formatted response text
    """
    if 'error' in graphrag_result:
        return graphrag_result['error']
    
    evidence = graphrag_result.get('evidence', {})
    response_parts = []
    
    # Format tasks
    tasks = evidence.get('tasks', [])
    if tasks:
        response_parts.append("📋 **Tasks found:**")
        for task in tasks:
            task_info = [f"• **{task['name']}**"]
            details = task['details']
            if details.get('due_date'):
                task_info.append(f"Due: {details['due_date']}")
            if details.get('owner'):
                task_info.append(f"Owner: {details['owner']}")
            if details.get('summary'):
                task_info.append(f"Summary: {details['summary']}")
            response_parts.append(" | ".join(task_info))
    
    # Format people
    people = evidence.get('people', [])
    if people:
        response_parts.append("\n👥 **People involved:**")
        for person in people:
            person_info = [f"• **{person['name']}**"]
            details = person['details']
            if details.get('role'):
                person_info.append(f"Role: {details['role']}")
            response_parts.append(" | ".join(person_info))
    
    # Format deadlines
    deadlines = evidence.get('deadlines', [])
    if deadlines:
        response_parts.append("\n📅 **Important dates:**")
        for deadline in deadlines:
            context = " | ".join(deadline['context']) if deadline['context'] else "Related activity"
            response_parts.append(f"• **{deadline['date']}**: {context}")
    
    if not response_parts:
        return "No specific information found for your query."
    
    # Add confidence indicator
    confidence = graphrag_result.get('confidence_score', 0.0)
    if confidence > 0.8:
        confidence_indicator = "🟢 High confidence"
    elif confidence > 0.5:
        confidence_indicator = "🟡 Medium confidence"
    else:
        confidence_indicator = "🔴 Low confidence"
    
    response_parts.append(f"\n{confidence_indicator} (score: {confidence})")
    
    return "\n".join(response_parts)
