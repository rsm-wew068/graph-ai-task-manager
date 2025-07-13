#!/usr/bin/env python3
"""
SIMPLE GraphRAG - No more complexity hell!
Just semantic search + graph expansion. That's it.
"""
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import math
import urllib.parse
from dotenv import load_dotenv
from typing import List, Dict, Tuple

load_dotenv()

class GraphRAG:
    """GraphRAG implementation using Neo4j for semantic search and graph traversal."""
    
    def __init__(self):
        """Initialize GraphRAG with Neo4j connection and sentence transformer."""
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
    
    def _get_topic_nodes(self) -> List[str]:
        """Get all topic nodes from Neo4j."""
        with self.driver.session() as session:
            result = session.run("MATCH (n:Topic) RETURN n.name AS name")
            return [record["name"] for record in result]
    
    def _get_node_labels(self, node_names: List[str]) -> Dict[str, str]:
        """Get labels for a list of node names."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n {name: $name}) RETURN n.name AS name, labels(n)[0] AS label",
                name=node_names[0] if node_names else ""
            )
            return {record["name"]: record["label"] for record in result}
    
    def _get_all_nodes(self) -> List[Dict]:
        """Get all nodes with their labels."""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN n.name AS name, labels(n)[0] AS label")
            return [{"name": record["name"], "label": record["label"]} for record in result]
    
    def _get_neighbors(self, node_name: str) -> List[str]:
        """Get neighboring nodes for a given node."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n {name: $name})--(m) RETURN m.name AS name",
                name=node_name
            )
            return [record["name"] for record in result]
    
    def _get_node_attrs(self, node_name: str) -> Dict:
        """Get all attributes of a node."""
        with self.driver.session() as session:
            result = session.run("MATCH (n {name: $name}) RETURN n, labels(n)[0] AS label", name=node_name)
            rec = result.single()
            if rec:
                node = rec["n"]
                label = rec["label"]
                return {"name": node["name"], "label": label, **dict(node)}
            return {}

    def _store_embeddings_in_neo4j(self):
        """Compute and store node embeddings as a property in Neo4j."""
        nodes = self._get_all_nodes()
        for node in nodes:
            label = node.get("label", "")
            name = node.get("name", "")
            text = f"{label}: {name}"
            embedding = self.embedder.encode(text).tolist()
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (n {name: $name})
                    SET n.vector = $vector
                    """,
                    name=name,
                    vector=embedding
                )

    def create_vector_index(self, label: str = "Topic", property: str = "vector", dims: int = 384):
        """Create a vector index in Neo4j for the given label/property if it doesn't exist."""
        with self.driver.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property}) OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, `vector.similarity_function`: 'cosine'}}}}
                """
            )

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Semantic search using Neo4j vector index."""
        query_embedding = self.embedder.encode(query).tolist()
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('Topic', 'vector', $topK, $embedding) YIELD node, score
                RETURN node.name AS name, score
                ORDER BY score DESC
                """,
                topK=top_k,
                embedding=query_embedding
            )
            return [(record["name"], record["score"]) for record in result]

    def expand_from_nodes(self, start_nodes: List[str], max_nodes: int = 10) -> set:
        # BFS expansion in Neo4j
        visited = set(start_nodes)
        queue = list(start_nodes)
        while queue and len(visited) < max_nodes:
            current = queue.pop(0)
            neighbors = self._get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < max_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _compute_embeddings(self):
        """Compute and store embeddings for all nodes in Neo4j."""
        try:
            self._store_embeddings_in_neo4j()
            # Create vector index if it doesn't exist
            self.create_vector_index()
            print("✅ Embeddings computed and stored in Neo4j")
        except Exception as e:
            print(f"⚠️ Warning: Could not compute embeddings: {e}")

    def query(self, query: str, max_nodes: int = 25) -> Dict:
        """Query the graph using semantic search and graph expansion."""
        try:
            # Ensure embeddings are computed
            self._compute_embeddings()
            
            # Perform semantic search
            topic_matches = self.semantic_search(query, top_k=5)
            good_topics = [topic for topic, score in topic_matches if score >= 0.5]
            
            if good_topics:
                all_nodes = self.expand_from_nodes(good_topics, max_nodes=max_nodes)
                confidence = topic_matches[0][1] if topic_matches else 0.0
                
                # Get detailed information about the nodes
                node_details = []
                for node_name in list(all_nodes)[:max_nodes]:
                    details = self._get_node_attrs(node_name)
                    if details:
                        node_details.append(details)
                
                return {
                    'query': query,
                    'relevant_nodes': [(topic, score) for topic, score in topic_matches if score >= 0.5],
                    'all_nodes': list(all_nodes),
                    'node_details': node_details,
                    'confidence_score': round(confidence, 3),
                    'explanation': f"Found {len(all_nodes)} nodes from {len(good_topics)} related topic(s)",
                    'method': 'neo4j_vector_search'
                }
            else:
                available_topics = self._get_topic_nodes()
                topic_list = ", ".join(available_topics[:10])  # Limit to first 10
                error_msg = f'No topic found matching "{query}". Available topics: {topic_list}.'
                return {
                    'query': query,
                    'error': error_msg,
                    'nodes': [],
                    'method': 'no_match'
                }
        except Exception as e:
            return {
                'query': query,
                'error': f'Query failed: {str(e)}',
                'nodes': [],
                'method': 'error'
            }

    def query_with_semantic_reasoning(self, query: str) -> Dict:
        """Enhanced query with semantic reasoning using Neo4j."""
        return self.query(query, max_nodes=30)

    def load_graph_with_embeddings(self) -> bool:
        """Load graph and compute embeddings. Returns True if successful."""
        try:
            # Check if we have any data in Neo4j
            topics = self._get_topic_nodes()
            if not topics:
                print("⚠️ No topics found in Neo4j database")
                return False
            
            # Compute embeddings
            self._compute_embeddings()
            print(f"✅ Loaded graph with {len(topics)} topics")
            return True
        except Exception as e:
            print(f"❌ Failed to load graph: {e}")
            return False

    def generate_visualization_html(self, query: str, result: Dict) -> str:
        """Generate enhanced visualization of the query results with multiple options."""
        if 'error' in result:
            return f"<p>Error: {result['error']}</p>"
        
        if not result.get('node_details'):
            return "<p>No data available for visualization</p>"
        
        # Get Neo4j Browser URL for direct graph exploration
        neo4j_browser_url = "http://localhost:7474"
        
        # Create Cypher query for the found nodes
        node_names = [node.get('name', '') for node in result.get('node_details', []) if node.get('name')]
        if node_names:
            # Create a Cypher query to visualize the subgraph
            cypher_query = f"""
            MATCH (n)
            WHERE n.name IN {node_names}
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE m.name IN {node_names}
            RETURN n, r, m
            """
            # URL encode the query for Neo4j Browser
            encoded_query = urllib.parse.quote(cypher_query.strip())
            neo4j_url = f"{neo4j_browser_url}/browser/?cmd={encoded_query}"
        else:
            neo4j_url = neo4j_browser_url
        
        # Create enhanced HTML with multiple visualization options
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-bottom: 15px;">🔍 Query Results: {query}</h3>
            
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <div style="flex: 1; background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4 style="color: #3498db; margin-top: 0;">📊 Metrics</h4>
                    <p><strong>Confidence:</strong> <span style="color: #e74c3c; font-weight: bold;">{result.get('confidence_score', 0):.3f}</span></p>
                    <p><strong>Method:</strong> {result.get('method', 'unknown')}</p>
                    <p><strong>Nodes Found:</strong> {len(result.get('node_details', []))}</p>
                </div>
                
                <div style="flex: 1; background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4 style="color: #27ae60; margin-top: 0;">🔗 Neo4j Browser</h4>
                    <p>Explore the full graph interactively:</p>
                    <a href="{neo4j_url}" target="_blank" style="background: #3498db; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block;">
                        🕸️ Open Neo4j Browser
                    </a>
                </div>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #8e44ad; margin-top: 0;">📋 Found Nodes ({len(result.get('node_details', []))})</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
        """
        
        # Group nodes by label for better organization
        nodes_by_label = {}
        for node in result.get('node_details', []):
            label = node.get('label', 'Unknown')
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(node.get('name', 'Unknown'))
        
        for label, names in nodes_by_label.items():
            html += f"""
                    <div style="border: 1px solid #ecf0f1; padding: 10px; border-radius: 4px;">
                        <h5 style="color: #2c3e50; margin: 0 0 8px 0;">{label} ({len(names)})</h5>
                        <ul style="margin: 0; padding-left: 20px;">
            """
            for name in names:
                html += f"<li style='margin-bottom: 4px;'>{name}</li>"
            html += """
                        </ul>
                    </div>
            """
        
        html += """
                </div>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background: #e8f4fd; border-radius: 4px; border-left: 4px solid #3498db;">
                <p style="margin: 0; font-size: 14px;">
                    <strong>💡 Tip:</strong> Click "Open Neo4j Browser" to explore the full graph interactively, 
                    run custom Cypher queries, and visualize relationships between nodes.
                </p>
            </div>
        </div>
        """
        
        return html

    def generate_plotly_visualization(self, query: str, result: Dict):
        """Generate interactive Plotly visualization of the graph."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            if 'error' in result or not result.get('node_details'):
                return None
            
            # Create node positions using a simple layout
            nodes = result.get('node_details', [])
            node_positions = {}
            
            # Simple circular layout
            radius = 200
            center_x, center_y = 0, 0
            
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / len(nodes)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                node_positions[node.get('name', '')] = (x, y)
            
            # Create node traces
            node_x = [pos[0] for pos in node_positions.values()]
            node_y = [pos[1] for pos in node_positions.values()]
            node_names = list(node_positions.keys())
            node_labels = [node.get('label', 'Unknown') for node in nodes]
            
            # Color nodes by label
            unique_labels = list(set(node_labels))
            colors = px.colors.qualitative.Set3[:len(unique_labels)]
            node_colors = [colors[unique_labels.index(label)] for label in node_labels]
            
            # Create the figure
            fig = go.Figure()
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='white')
                ),
                text=node_names,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{text}</b><br>Label: %{customdata}<extra></extra>',
                customdata=node_labels,
                name='Nodes'
            ))
            
            # Add edges (simplified - just show connections between nodes)
            edge_x = []
            edge_y = []
            
            # Get relationships from Neo4j
            with self.driver.session() as session:
                for node_name in node_names:
                    result_rel = session.run(
                        """
                        MATCH (n {name: $name})-[r]-(m)
                        WHERE m.name IN $node_names
                        RETURN n.name as source, m.name as target, type(r) as relationship
                        """,
                        name=node_name,
                        node_names=node_names
                    )
                    
                    for record in result_rel:
                        source = record["source"]
                        target = record["target"]
                        if source in node_positions and target in node_positions:
                            edge_x.extend([node_positions[source][0], node_positions[target][0], None])
                            edge_y.extend([node_positions[source][1], node_positions[target][1], None])
            
            if edge_x:  # Only add edges if we have any
                fig.add_trace(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Graph Visualization: {query}",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=500
            )
            
            return fig
            
        except ImportError:
            return None
        except Exception as e:
            print(f"Error generating Plotly visualization: {e}")
            return None

    def generate_pyvis_visualization(self, query: str, result: Dict) -> str:
        """Generate interactive Pyvis network visualization of the graph."""
        try:
            from pyvis.network import Network
            import tempfile
            
            if 'error' in result or not result.get('node_details'):
                return "<p>No data available for visualization</p>"
            
            # Create network
            net = Network(
                height="500px", 
                width="100%", 
                bgcolor="#ffffff", 
                font_color="#2c3e50",
                notebook=False, 
                directed=False,
                select_menu=True,
                filter_menu=True
            )
            
            # Configure physics for better layout
            net.set_options("""
            var options = {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "solver": "forceAtlas2Based",
                "timestep": 0.35
              }
            }
            """)
            
            # Color scheme for different node types
            label_colors = {
                "Task": "#FF6B6B",      # Red
                "Person": "#4ECDC4",    # Teal
                "Topic": "#FFD93D",     # Yellow
                "Date": "#1A535C",      # Dark teal
                "Email": "#FFB400",     # Orange
                "Summary": "#B5EAD7",   # Light green
                "Role": "#B2A4FF",      # Purple
                "Department": "#FFB7B2", # Light red
                "Organization": "#C7CEEA", # Light blue
                "Unknown": "#CCCCCC"    # Gray
            }
            
            # Add nodes
            for node in result.get('node_details', []):
                label = node.get('label', 'Unknown')
                name = node.get('name', 'Unknown')
                color = label_colors.get(label, "#CCCCCC")
                
                # Create tooltip with node details
                tooltip = f"<b>{label}</b><br>Name: {name}"
                if 'email_id' in node:
                    tooltip += f"<br>Email ID: {node['email_id']}"
                if 'due_date' in node:
                    tooltip += f"<br>Due Date: {node['due_date']}"
                if 'start_date' in node:
                    tooltip += f"<br>Start Date: {node['start_date']}"
                if 'topic' in node:
                    tooltip += f"<br>Topic: {node['topic']}"
                if 'role' in node:
                    tooltip += f"<br>Role: {node['role']}"
                if 'department' in node:
                    tooltip += f"<br>Department: {node['department']}"
                if 'organization' in node:
                    tooltip += f"<br>Organization: {node['organization']}"
                if 'relationship_type' in node:
                    tooltip += f"<br>Type: {node['relationship_type'].title()}"
                if 'date_type' in node:
                    tooltip += f"<br>Date Type: {node['date_type']}"
                
                net.add_node(
                    name, 
                    label=f"{name}", 
                    color=color,
                    title=tooltip,
                    size=25 if label == "Task" else 20  # Make tasks slightly larger
                )
            
            # Add edges (relationships)
            node_names = [node.get('name', '') for node in result.get('node_details', [])]
            edge_count = 0
            
            with self.driver.session() as session:
                for node_name in node_names:
                    result_rel = session.run(
                        """
                        MATCH (n {name: $name})-[r]-(m)
                        WHERE m.name IN $node_names
                        RETURN n.name as source, m.name as target, type(r) as relationship
                        """,
                        name=node_name,
                        node_names=node_names
                    )
                    
                    for record in result_rel:
                        source = record["source"]
                        target = record["target"]
                        rel = record["relationship"]
                        
                        # Color edges by relationship type
                        edge_color = "#666666"  # Default gray
                        if "RESPONSIBLE_TO" in rel:
                            edge_color = "#e74c3c"  # Red for responsibility
                        elif "COLLABORATED_BY" in rel:
                            edge_color = "#3498db"  # Blue for collaboration
                        elif "HAS_TASK" in rel:
                            edge_color = "#f39c12"  # Orange for task ownership
                        elif "DUE_ON" in rel:
                            edge_color = "#27ae60"  # Green for dates
                        
                        net.add_edge(
                            source, 
                            target, 
                            label=rel.replace("_", " ").title(),
                            color=edge_color,
                            width=2
                        )
                        edge_count += 1
            
            # Add title and stats
            title_html = f"""
            <div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #2c3e50;">🔍 Graph for: {query}</h3>
                <p style="margin: 5px 0; color: #7f8c8d; font-size: 14px;">
                    📊 {len(result.get('node_details', []))} nodes • {edge_count} relationships
                </p>
            </div>
            """
            
            # Generate HTML
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w') as tmp_file:
                net.show(tmp_file.name)
                tmp_file.seek(0)
                html_content = tmp_file.read()
            
            # Combine title with network HTML
            full_html = title_html + html_content
            
            return full_html
            
        except ImportError:
            return "<p>Pyvis not installed. Install with: pip install pyvis</p>"
        except Exception as e:
            print(f"Error generating Pyvis visualization: {e}")
            return f"<p>Error generating visualization: {str(e)}</p>"

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()


def format_graphrag_response(result: Dict) -> str:
    """Format GraphRAG response for LLM consumption."""
    if 'error' in result:
        return f"Error: {result['error']}"
    
    if not result.get('node_details'):
        return "No relevant information found."
    
    # Format the response
    response = f"Query: {result.get('query', 'Unknown')}\n"
    response += f"Confidence: {result.get('confidence_score', 0):.3f}\n"
    response += f"Method: {result.get('method', 'unknown')}\n\n"
    
    response += "Found Information:\n"
    for node in result.get('node_details', []):
        label = node.get('label', 'Unknown')
        name = node.get('name', 'Unknown')
        response += f"- {label}: {name}\n"
    
    return response
