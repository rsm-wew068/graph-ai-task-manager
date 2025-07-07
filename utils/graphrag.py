#!/usr/bin/env python3
"""
SIMPLE GraphRAG - No more complexity hell!
Just semantic search + graph expansion. That's it.
"""
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np


class GraphRAG:
    """Dead simple GraphRAG - find stuff, expand from there."""
    
    def __init__(self, graph_path: str = "/tmp/topic_graph.gpickle"):
        self.graph_path = graph_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph: Optional[nx.DiGraph] = None
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
    def load_graph_with_embeddings(self) -> bool:
        """Load graph and compute semantic embeddings."""
        if not os.path.exists(self.graph_path):
            return False
            
        try:
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)
            self._compute_embeddings()
            return True
        except Exception:
            return False
    
    def _compute_embeddings(self):
        """Compute AI embeddings for all nodes."""
        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get("label", "")
            name = attrs.get("name", str(node))
            text = f"{label}: {name}"
            
            embedding = self.embedder.encode(text)
            self.node_embeddings[node] = embedding
    def semantic_search(self, query: str,
                        top_k: int = 10) -> List[Tuple[str, float]]:
        """Find nodes most similar to query."""
        if not self.node_embeddings:
            return []
            
        query_embedding = self.embedder.encode(query)
        similarities = []
        
        for node, embedding in self.node_embeddings.items():
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            if sim >= 0.2:  # Reasonable threshold
                similarities.append((node, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def expand_from_nodes(self, start_nodes: List[str], max_nodes: int = 10, direct_only: bool = False) -> set:
        """Expand from starting nodes following only relevant connections."""
        connected = set(start_nodes)
        
        if direct_only:
            # Only add direct neighbors - no expansion of expansions
            for node in start_nodes:
                if node not in self.graph:
                    continue
                    
                # Add direct neighbors (outgoing)
                for neighbor in self.graph.neighbors(node):
                    connected.add(neighbor)
                    
                # Add direct predecessors (incoming)
                for predecessor in self.graph.predecessors(node):
                    connected.add(predecessor)
            
            return connected
        
        # Topic-centered expansion: only expand through meaningful relationships
        to_expand = list(start_nodes)
        
        while to_expand and len(connected) < max_nodes:
            current_node = to_expand.pop(0)
            if current_node not in self.graph:
                continue
                
            current_attrs = self.graph.nodes.get(current_node, {})
            current_label = current_attrs.get('label', '')
            
            # Add directly connected nodes based on meaningful relationships
            for neighbor in self.graph.neighbors(current_node):
                if len(connected) >= max_nodes:
                    break
                if neighbor not in connected:
                    neighbor_attrs = self.graph.nodes.get(neighbor, {})
                    neighbor_label = neighbor_attrs.get('label', '')
                    
                    # Get the edge relationship
                    edge_data = self.graph.get_edge_data(current_node, neighbor, {})
                    edge_label = edge_data.get('label', '')
                    
                    # Only include nodes with strong semantic relationships
                    should_include = False
                    
                    if current_label == 'Topic':
                        # From Topic: include tasks directly labeled with this topic
                        if neighbor_label == 'Task' and edge_label == 'HAS_TASK':
                            should_include = True
                            
                    elif current_label == 'Task':
                        # From Task: include assignees, dates, summaries - NOT other tasks
                        if neighbor_label in ['Person', 'Date', 'Summary', 'Email Index'] and edge_label in ['RESPONSIBLE_TO', 'COLLABORATED_BY', 'DUE_ON', 'START_ON', 'BASED_ON', 'LINKED_TO']:
                            should_include = True
                            
                    elif current_label == 'Person':
                        # From Person: include their role/department/organization hierarchy
                        if neighbor_label in ['Role', 'Department', 'Organization'] and edge_label in ['HAS_ROLE', 'BELONGS_TO', 'IS_IN']:
                            should_include = True
                            
                    elif current_label == 'Role':
                        # From Role: include department
                        if neighbor_label == 'Department' and edge_label == 'BELONGS_TO':
                            should_include = True
                            
                    elif current_label == 'Department':
                        # From Department: include organization
                        if neighbor_label == 'Organization' and edge_label == 'IS_IN':
                            should_include = True
                    
                    if should_include:
                        connected.add(neighbor)
                        
                        # Queue for expansion to get full hierarchies
                        if neighbor_label in ['Task', 'Person', 'Role', 'Department']:
                            to_expand.append(neighbor)
            
            # Also check predecessors for reverse relationships
            for predecessor in self.graph.predecessors(current_node):
                if len(connected) >= max_nodes:
                    break
                if predecessor not in connected:
                    pred_attrs = self.graph.nodes.get(predecessor, {})
                    pred_label = pred_attrs.get('label', '')
                    
                    # Get the edge relationship
                    edge_data = self.graph.get_edge_data(predecessor, current_node, {})
                    edge_label = edge_data.get('label', '')
                    
                    # Include meaningful reverse relationships
                    should_include = False
                    
                    if current_label == 'Task' and pred_label == 'Topic' and edge_label == 'HAS_TASK':
                        should_include = True
                    elif current_label in ['Date', 'Summary', 'Email Index'] and pred_label == 'Task':
                        should_include = True
                    elif current_label in ['Role', 'Department', 'Organization'] and pred_label == 'Person':
                        should_include = True
                    
                    if should_include:
                        connected.add(predecessor)
        
        return connected
    
    def query(self, query: str, direct_only: bool = False) -> Dict:
        """Topic-centered query for maximum accuracy."""
        if not self.load_graph_with_embeddings():
            return {
                'query': query,
                'error': 'No graph found. Process emails first.',
                'nodes': []
            }
        
        # Step 1: Try topic name matching first (highest accuracy)
        topic_matches = self.search_topics_by_name(query, semantic_threshold=0.5)
        
        if topic_matches:
            # Found topic name matches - use only the BEST match for topic-centered approach
            best_topic = topic_matches[0]  # Take only the highest scoring topic
            start_nodes = [best_topic[0]]  # Single topic only
            all_nodes = self.expand_from_nodes(
                start_nodes, max_nodes=15, direct_only=direct_only
            )
            confidence = best_topic[1]
            
            explanation = f"Found {len(all_nodes)} nodes from topic '{best_topic[0]}'"
            if direct_only:
                explanation += " (direct neighbors only)"
            
            return {
                'query': query,
                'relevant_nodes': [best_topic],  # Only the best topic
                'all_nodes': list(all_nodes),
                'confidence_score': round(confidence, 3),
                'explanation': explanation,
                'method': 'topic_name_search'
            }
        
        # No topic matches found - show actual topics
        available_topics = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('label') == 'Topic':
                topic_name = attrs.get('name', str(node))
                available_topics.append(topic_name)
        
        if available_topics:
            topic_list = ", ".join(available_topics)
            error_msg = f'No topic found matching "{query}". Available: {topic_list}.'
        else:
            error_msg = f'No topic found matching "{query}". No topics in graph.'
        
        return {
            'query': query,
            'error': error_msg,
            'nodes': [],
            'method': 'no_match'
        }
    
    def generate_visualization_html(self, query: str, result: Dict) -> str:
        """Generate visualization HTML content directly without saving to file."""
        try:
            from pyvis.network import Network
        except ImportError:
            return "<p>pyvis not installed</p>"
        
        if not self.graph:
            return "<p>No graph loaded</p>"
        
        try:
            net = Network(height="600px", width="100%")
            
            # Show only connected nodes
            nodes_to_show = set(result.get('all_nodes', []))
            if not nodes_to_show:
                return "<p>No nodes found in query result</p>"
                
            subgraph = self.graph.subgraph(nodes_to_show)
            
            # Colors for topic-centered hierarchy
            colors = {
                'Topic': '#FF6B9D',      # Pink - most important
                'Task': '#90EE90',       # Light green
                'Person': '#87CEEB',     # Sky blue  
                'Role': '#FFA500',       # Orange
                'Department': '#DDA0DD', # Plum
                'Organization': '#F0E68C', # Khaki
                'Date': '#D3D3D3',       # Light gray
                'Summary': '#FFE4B5',    # Moccasin
                'Email Index': '#E6E6FA' # Lavender
            }
            
            # Add nodes with topic-centered sizing
            for node, attrs in subgraph.nodes(data=True):
                label = attrs.get('label', '')
                name = attrs.get('name', str(node))
                color = colors.get(label, '#BDC3C7')
                
                # Topic-centered node sizing
                if label == 'Topic':
                    node_size = 50  # Largest - center of graph
                elif label == 'Task':
                    node_size = 35  # Second largest
                elif label == 'Person':
                    node_size = 25  # Medium
                else:
                    node_size = 20  # Smaller for support nodes
                
                # Simple display names - no embedding role in Person labels
                if label == 'Task':
                    display_name = name[:40] + "..." if len(name) > 40 else name
                elif label == 'Summary':
                    display_name = name[:50] + "..." if len(name) > 50 else name
                else:
                    display_name = name[:30] + "..." if len(name) > 30 else name
                
                # Create detailed tooltip with all attributes
                tooltip_parts = [f"<b>{label}</b>: {name}"]
                for key, value in attrs.items():
                    if key not in ['label', 'name'] and value:
                        tooltip_parts.append(f"{key}: {value}")
                
                # For Person nodes, add FULL role/dept/org info to tooltip
                if label == 'Person':
                    person_details = _get_person_details(self.graph, node)
                    if person_details:
                        details_clean = person_details.strip('() ')
                        tooltip_parts.append(f"<b>Full Details:</b> {details_clean}")
                
                tooltip = "<br>".join(tooltip_parts)
                
                net.add_node(
                    node,
                    label=display_name,
                    title=tooltip,
                    color=color,
                    size=node_size,
                    font={'size': 14, 'color': 'black'}
                )
            
            # Add edges
            for u, v, edge_attrs in subgraph.edges(data=True):
                edge_label = edge_attrs.get('label', '')
                net.add_edge(u, v, label=edge_label)
            
            # Set heading and generate HTML
            net.heading = f"Query: {query}"
            
            # Generate HTML content directly
            html_content = net.generate_html()
            return html_content
            
        except Exception as e:
            return f"<p>Error generating visualization: {str(e)}</p>"

    def search_topics_by_name(self, query: str, semantic_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Search for topics using semantic similarity for higher accuracy."""
        if not self.graph or not self.node_embeddings:
            return []
        
        # Encode the query
        query_embedding = self.embedder.encode(query)
        topic_matches = []
        
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('label') == 'Topic':
                # Get the embedding for this topic node
                if node in self.node_embeddings:
                    topic_embedding = self.node_embeddings[node]
                    
                    # Calculate semantic similarity
                    similarity = cosine_similarity([query_embedding], [topic_embedding])[0][0]
                    
                    # Only include topics with good semantic similarity
                    if similarity >= semantic_threshold:
                        topic_matches.append((node, similarity))
        
        return sorted(topic_matches, key=lambda x: x[1], reverse=True)

    # Compatibility methods
    def query_with_semantic_reasoning(self, query: str) -> Dict:
        return self.query(query)


def format_response(result: Dict) -> str:
    """Format response like the old system with structured task details."""
    if 'error' in result:
        return result['error']
    
    if not result.get('all_nodes'):
        return "No information found."
    
    try:
        import pickle
        with open("/tmp/topic_graph.gpickle", "rb") as f:
            graph = pickle.load(f)
        
        # Find all tasks in the result
        tasks = []
        for node in result.get('all_nodes', []):
            if node in graph:
                attrs = graph.nodes[node]
                if attrs.get('label') == 'Task':
                    tasks.append(node)
        
        if not tasks:
            return "No tasks found in the results."
        
        # Format each task in the structured format
        response_parts = []
        
        for task_node in tasks:
            task_attrs = graph.nodes[task_node]
            task_name = task_attrs.get('name', str(task_node))
            
            task_info = [f"**Task:** {task_name}"]
            
            # Find the topic for this task
            for neighbor in graph.neighbors(task_node):
                edge_data = graph.get_edge_data(task_node, neighbor, {})
                edge_label = edge_data.get('label', '')
                neighbor_attrs = graph.nodes[neighbor]
                
                if neighbor_attrs.get('label') == 'Topic':
                    topic_name = neighbor_attrs.get('name', neighbor)
                    task_info.append(f"**Topic:** {topic_name}")
                    break
            
            # Get all the direct neighbors with their relationships
            for neighbor in graph.neighbors(task_node):
                edge_data = graph.get_edge_data(task_node, neighbor, {})
                edge_label = edge_data.get('label', '')
                neighbor_attrs = graph.nodes[neighbor]
                neighbor_name = neighbor_attrs.get('name', neighbor)
                neighbor_label = neighbor_attrs.get('label', '')
                
                if edge_label == 'START_ON':
                    task_info.append(f"   • **Start Date:** {neighbor_name}")
                elif edge_label == 'DUE_ON':
                    task_info.append(f"   • **Due Date:** {neighbor_name}")
                elif edge_label == 'BASED_ON' or neighbor_label == 'Summary':
                    task_info.append(f"   • **Summary:** {neighbor_name}")
                elif edge_label == 'LINKED_TO' or neighbor_label == 'Email Index':
                    task_info.append(f"   • **Email Index:** {neighbor_name}")
                elif edge_label == 'RESPONSIBLE_TO':
                    # Get role/dept/org info for the person
                    person_details = _get_person_details(graph, neighbor)
                    task_info.append(f"   • **Responsible To:** {neighbor_name}{person_details}")
                elif edge_label == 'COLLABORATED_BY':
                    person_details = _get_person_details(graph, neighbor)
                    task_info.append(f"   • **Collaborated By:** {neighbor_name}{person_details}")
            
            response_parts.append("\n".join(task_info))
        
        # Add confidence at the end
        confidence = result.get('confidence_score', 0.0)
        conf_text = "🟢 High" if confidence > 0.7 else "🟡 Medium" if confidence > 0.4 else "🔴 Low"
        response_parts.append(f"\n**Confidence:** {conf_text} ({confidence})")
        
        return "\n\n".join(response_parts)
        
    except Exception as e:
        return f"📊 Error formatting response: {str(e)}"

def _get_person_details(graph, person_node):
    """Get role, department, organization details for a person."""
    details = []
    
    for neighbor in graph.neighbors(person_node):
        edge_data = graph.get_edge_data(person_node, neighbor, {})
        edge_label = edge_data.get('label', '')
        neighbor_attrs = graph.nodes[neighbor]
        
        if edge_label == 'HAS_ROLE' or neighbor_attrs.get('label') == 'Role':
            role_name = neighbor_attrs.get('name', neighbor)
            details.append(f"Role: {role_name}")
            
            # Get department for this role
            for dept_neighbor in graph.neighbors(neighbor):
                dept_edge = graph.get_edge_data(neighbor, dept_neighbor, {})
                dept_edge_label = dept_edge.get('label', '')
                dept_attrs = graph.nodes[dept_neighbor]
                
                if dept_edge_label == 'BELONGS_TO' or dept_attrs.get('label') == 'Department':
                    dept_name = dept_attrs.get('name', dept_neighbor)
                    details.append(f"Department: {dept_name}")
                    
                    # Get organization for this department
                    for org_neighbor in graph.neighbors(dept_neighbor):
                        org_edge = graph.get_edge_data(dept_neighbor, org_neighbor, {})
                        org_edge_label = org_edge.get('label', '')
                        org_attrs = graph.nodes[org_neighbor]
                        
                        if org_edge_label == 'IS_IN' or org_attrs.get('label') == 'Organization':
                            org_name = org_attrs.get('name', org_neighbor)
                            details.append(f"Organization: {org_name}")
                            break
                    break
            break
    
    if details:
        return f" ({', '.join(details)})"
    return ""


def format_graphrag_response(result: Dict) -> str:
    """Compatibility function."""
    return format_response(result)
