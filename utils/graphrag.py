#!/usr/bin/env python3
"""
Simplified GraphRAG - focused on what actually matters.
~200 lines instead of 800.
"""
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np


class GraphRAG:
    """Simplified GraphRAG - AI-powered graph querying without unnecessary complexity."""
    
    def __init__(self, graph_path: str = "topic_graph.gpickle"):
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
            text = f"{label}: {name}"  # Simple: "Person: Rachel Martinez"
            
            embedding = self.embedder.encode(text)
            self.node_embeddings[node] = embedding
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find nodes most similar to the query with improved keyword extraction."""
        if not self.node_embeddings:
            return []
            
        # Extract key terms from query, ignoring common question words
        stop_words = {'what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 
                     'before', 'after', 'above', 'below', 'between', 'among', 'task', 
                     'tasks', 'related', 'find', 'show', 'get', 'tell', 'me'}
        
        query_words = [word.strip().lower() for word in query.split() 
                      if word.strip().lower() not in stop_words and len(word.strip()) > 2]
        key_terms = ' '.join(query_words)
        
        query_embedding = self.embedder.encode(query)
        
        similarities = []
        exact_matches = []
        
        for node, embedding in self.node_embeddings.items():
            # Get node text
            attrs = self.graph.nodes.get(node, {})
            label = attrs.get("label", "")
            name = attrs.get("name", str(node))
            text = f"{label}: {name}".lower()
            
            # Check for exact keyword matches with key terms
            if key_terms and (key_terms in text or 
                             any(term in text for term in query_words if len(term) > 3)):
                exact_matches.append((node, 1.0))  # Perfect match
                
            # Also compute semantic similarity
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            if sim >= 0.15:  # Lower threshold for more matches
                similarities.append((node, sim))
        
        # Prioritize exact matches, then semantic matches
        all_matches = exact_matches + similarities
        
        # Remove duplicates and sort by score
        seen = set()
        unique_matches = []
        for node, score in all_matches:
            if node not in seen:
                seen.add(node)
                unique_matches.append((node, score))
        
        return sorted(unique_matches, key=lambda x: x[1], reverse=True)[:top_k]
    
    def find_connected_nodes(self, start_nodes: List[str], max_nodes: int = 10) -> set:
        """Find nodes connected to start nodes following the schema paths SELECTIVELY."""
        connected = set(start_nodes)
        
        # Follow schema relationships selectively - check both incoming and outgoing edges
        for node in start_nodes:
            if node not in self.graph:
                continue
                
            # Check OUTGOING edges (node -> neighbor)
            for neighbor in self.graph.neighbors(node):
                if len(connected) >= max_nodes:
                    break
                    
                edge_data = self.graph.get_edge_data(node, neighbor, {})
                edge_label = edge_data.get('label', '')
                neighbor_attrs = self.graph.nodes.get(neighbor, {})
                neighbor_label = neighbor_attrs.get('label', '')
                
                # Only add neighbors that are Tasks or form meaningful task-related paths
                if neighbor_label == 'Task':
                    connected.add(neighbor)
                    self._add_task_context(neighbor, connected, max_nodes)
                elif edge_label in ['HAS_TASK', 'RESPONSIBLE_TO', 'START_ON', 'DUE_ON', 'BASED_ON', 'LINKED_TO']:
                    connected.add(neighbor)
                    
                    # If this is a task, get its context
                    if edge_label == 'HAS_TASK':
                        self._add_task_context(neighbor, connected, max_nodes)
                    # If this is a person, get their role/dept/org
                    elif edge_label in ['RESPONSIBLE_TO', 'COLLABORATED_BY']:
                        self._add_person_context(neighbor, connected, max_nodes)
                elif edge_label in ['RESPONSIBLE_TO', 'COLLABORATED_BY']:
                    self._add_person_context(neighbor, connected, max_nodes)
            
            # Check INCOMING edges (neighbor -> node) - this is what was missing!
            for neighbor in self.graph.predecessors(node):
                if len(connected) >= max_nodes:
                    break
                    
                edge_data = self.graph.get_edge_data(neighbor, node, {})
                edge_label = edge_data.get('label', '')
                neighbor_attrs = self.graph.nodes.get(neighbor, {})
                neighbor_label = neighbor_attrs.get('label', '')
                
                # If a Task is responsible to this person, add the task
                if neighbor_label == 'Task' and edge_label == 'RESPONSIBLE_TO':
                    connected.add(neighbor)
                    self._add_task_context(neighbor, connected, max_nodes)
                # If this is a task pointing to this node, add it
                elif neighbor_label == 'Task':
                    connected.add(neighbor)
                    self._add_task_context(neighbor, connected, max_nodes)
        
        return connected
    
    def _add_task_context(self, task_node: str, connected: set, max_nodes: int):
        """Add task-related context: dates, summaries, owners."""
        if len(connected) >= max_nodes or task_node not in self.graph:
            return
            
        for neighbor in self.graph.neighbors(task_node):
            if len(connected) >= max_nodes:
                break
            edge_data = self.graph.get_edge_data(task_node, neighbor, {})
            edge_label = edge_data.get('label', '')
            
            # Follow schema: Task->Date, Task->Summary, Task->Person
            if edge_label in ['START_ON', 'DUE_ON', 'BASED_ON', 'LINKED_TO', 'RESPONSIBLE_TO']:
                connected.add(neighbor)
                
                # If person, add their organizational context
                neighbor_attrs = self.graph.nodes.get(neighbor, {})
                if neighbor_attrs.get('label') == 'Person':
                    self._add_person_context(neighbor, connected, max_nodes)
    
    def _add_person_context(self, person_node: str, connected: set, max_nodes: int):
        """Add person organizational context: role->dept->org."""
        if len(connected) >= max_nodes or person_node not in self.graph:
            return
            
        # Follow schema: Person->Role->Department->Organization
        for role_node in self.graph.neighbors(person_node):
            if len(connected) >= max_nodes:
                break
            edge_data = self.graph.get_edge_data(person_node, role_node, {})
            if edge_data.get('label') == 'HAS_ROLE':
                connected.add(role_node)
                
                for dept_node in self.graph.neighbors(role_node):
                    if len(connected) >= max_nodes:
                        break
                    edge_data = self.graph.get_edge_data(role_node, dept_node, {})
                    if edge_data.get('label') == 'BELONGS_TO':
                        connected.add(dept_node)
                        
                        for org_node in self.graph.neighbors(dept_node):
                            if len(connected) >= max_nodes:
                                break
                            edge_data = self.graph.get_edge_data(dept_node, org_node, {})
                            if edge_data.get('label') == 'IS_IN':
                                connected.add(org_node)
    
    def query(self, query: str) -> Dict:
        """Main query function - simplified."""
        if not self.load_graph_with_embeddings():
            return {
                'query': query,
                'error': 'No graph found. Process emails first.',
                'nodes': []
            }
        
        # Step 1: Find semantically relevant nodes
        relevant_nodes = self.semantic_search(query, top_k=3)
        
        if not relevant_nodes:
            return {
                'query': query,
                'error': 'No relevant information found.',
                'nodes': []
            }
        
        # Step 2: Find connected nodes
        start_nodes = [node for node, _ in relevant_nodes]
        all_nodes = self.find_connected_nodes(start_nodes, max_nodes=8)
        
        # Step 3: Calculate confidence based on semantic similarity
        confidence = max(score for _, score in relevant_nodes) if relevant_nodes else 0.0
        
        # Step 4: Return comprehensive response
        return {
            'query': query,
            'relevant_nodes': relevant_nodes,
            'all_nodes': list(all_nodes),
            'confidence_score': round(confidence, 3),
            'explanation': f"Found {len(relevant_nodes)} relevant nodes, expanded to {len(all_nodes)} total"
        }
    
    def visualize(self, query: str, result: Dict, output_path: str) -> str:
        """Create visualization - simplified."""
        try:
            from pyvis.network import Network
        except ImportError:
            return "pyvis not installed"
        
        if not self.graph:
            return "No graph loaded"
        
        try:
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            
            # Get nodes to visualize - only show connected nodes
            nodes_to_show = set(result.get('all_nodes', []))
            relevant_node_ids = {node for node, _ in result.get('relevant_nodes', [])}
            
            # Create subgraph and only keep nodes that have connections
            subgraph = self.graph.subgraph(nodes_to_show)
            
            # Filter out isolated nodes (nodes with no edges in the subgraph)
            connected_nodes = set()
            for node in subgraph.nodes():
                if subgraph.degree(node) > 0:  # Has at least one edge
                    connected_nodes.add(node)
            
            # If no connected nodes, keep at least the relevant ones
            if not connected_nodes:
                connected_nodes = relevant_node_ids
            
            # Create final subgraph with only connected nodes
            final_subgraph = subgraph.subgraph(connected_nodes)
            
            # Color map - simplified from old app
            colors = {
                'Task': '#90EE90',      # Light green
                'Person': '#87CEEB',    # Light sky blue  
                'Topic': '#FFB6C1',     # Light pink
                'Role': '#FFA500',      # Orange
                'Department': '#9370DB', # Purple
                'Organization': '#FA8072', # Salmon
                'Date': '#D3D3D3',      # Light gray
                'Summary': '#D3D3D3'    # Light gray
            }
            
            # Add nodes - only connected ones
            for node, attrs in final_subgraph.nodes(data=True):
                label = attrs.get('label', '')
                name = attrs.get('name', str(node))
                
                # Color based on node type, size based on relevance
                color = colors.get(label, '#BDC3C7')
                
                if node in relevant_node_ids:
                    size = 30  # Larger for most relevant
                else:
                    size = 25 if label in ['Task', 'Person', 'Topic'] else 15
                
                net.add_node(
                    node,
                    label=name[:30] + "..." if len(name) > 30 else name,
                    title=f"{label}: {name}",
                    color=color,
                    size=size
                )
            
            # Add edges - only between connected nodes
            for u, v, edge_attrs in final_subgraph.edges(data=True):
                edge_label = edge_attrs.get('label', '')
                net.add_edge(u, v, label=edge_label, color="#34495E", width=2)
            
            # Configure and save
            net.set_options('{"physics": {"enabled": true, "stabilization": {"iterations": 100}}}')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            net.heading = f"GraphRAG Query: {query}"
            net.save_graph(output_path)
            
            return output_path
            
        except Exception as e:
            return f"Visualization error: {str(e)}"
    
    def query_with_semantic_reasoning(self, query: str) -> Dict:
        """Compatibility method - same as query() but matches old API."""
        return self.query(query)
    
    def visualize_query_results(
        self, 
        query: str, 
        result: Dict, 
        output_path: str
    ) -> str:
        """Compatibility method - same as visualize() but matches old API."""
        return self.visualize(query, result, output_path)


def format_response(result: Dict) -> str:
    """Format response into structured data that GPT can understand."""
    if 'error' in result:
        return result['error']
    
    if not result.get('all_nodes'):
        return "No information found for your query."
    
    # Create structured response showing exactly what's in the graph
    response_parts = []
    
    # Load graph to get detailed node information
    try:
        import pickle
        with open("topic_graph.gpickle", "rb") as f:
            graph = pickle.load(f)
        
        # Categorize nodes by type for clear presentation
        tasks = []
        people = []
        dates = []
        topics = []
        
        for node in result.get('all_nodes', []):
            if node in graph:
                attrs = graph.nodes[node]
                label = attrs.get('label', '')
                name = attrs.get('name', str(node))
                
                if label == 'Task':
                    # Get task details with dates, owners, and collaborators
                    task_info = {'name': name, 'dates': [], 'owners': [], 'collaborators': []}
                    for neighbor in graph.neighbors(node):
                        edge_data = graph.get_edge_data(node, neighbor, {})
                        edge_label = edge_data.get('label', '')
                        neighbor_attrs = graph.nodes[neighbor]
                        neighbor_name = neighbor_attrs.get('name', neighbor)
                        
                        if edge_label in ['START_ON', 'DUE_ON']:
                            task_info['dates'].append(f"{edge_label}: {neighbor_name}")
                        elif edge_label == 'RESPONSIBLE_TO':
                            task_info['owners'].append(neighbor_name)
                        elif edge_label == 'COLLABORATED_BY':
                            task_info['collaborators'].append(neighbor_name)
                    
                    tasks.append(task_info)
                    
                elif label == 'Person':
                    people.append(name)
                elif label == 'Date':
                    dates.append(name)
                elif label == 'Topic':
                    topics.append(name)
        
        # Format structured output
        if tasks:
            response_parts.append("📋 **Tasks Found:**")
            for task in tasks:
                task_line = f"• {task['name']}"
                if task['dates']:
                    task_line += f" | {', '.join(task['dates'])}"
                if task['owners']:
                    task_line += f" | Owner: {', '.join(task['owners'])}"
                if task['collaborators']:
                    task_line += f" | Collaborators: {', '.join(task['collaborators'])}"
                response_parts.append(task_line)
        
        if people:
            response_parts.append(f"\n👥 **People Involved:** {', '.join(people)}")
        
        if dates:
            response_parts.append(f"\n📅 **Dates Found:** {', '.join(dates)}")
            
        if topics:
            response_parts.append(f"\n🏷️ **Topics:** {', '.join(topics)}")
        
        # Add confidence
        confidence = result.get('confidence_score', 0.0)
        confidence_text = "🟢 High" if confidence > 0.7 else "🟡 Medium" if confidence > 0.4 else "🔴 Low"
        response_parts.append(f"\n**Confidence:** {confidence_text} ({confidence})")
        
        return "\n".join(response_parts)
        
    except Exception:
        # Fallback to simple response if detailed analysis fails
        explanation = result.get('explanation', '')
        return f"📊 {explanation}\n\nFound relevant information in the knowledge graph."


def format_graphrag_response(result: Dict) -> str:
    """Compatibility function - same as format_response() but matches old API."""
    return format_response(result)


# Simple usage example:
# rag = GraphRAG()
# result = rag.query("Who is Rachel?")
# response_text = format_response(result)
# viz_path = rag.visualize("Who is Rachel?", result, "static/simple_viz.html")
