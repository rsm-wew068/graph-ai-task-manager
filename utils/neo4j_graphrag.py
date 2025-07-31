#!/usr/bin/env python3
"""
Neo4j GraphRAG - Replaces NetworkX with Neo4j for better performance!
Maintains the same interface as the original GraphRAG system.
"""
from openai import OpenAI
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

# Handle dotenv import gracefully for deployment environments
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class Neo4jGraphRAG:
    """Neo4j-based GraphRAG system for intelligent task querying."""
    
    def __init__(self, 
                 neo4j_uri: str = None, 
                 neo4j_username: str = None,
                 neo4j_password: str = None,
                 embedding_model: str = "text-embedding-3-small"):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.openai_client = OpenAI(api_key=api_key)
        
        self.driver = None
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
    def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_username, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Connected to Neo4j for GraphRAG")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def load_graph_with_embeddings(self) -> bool:
        """Load graph data and compute semantic embeddings."""
        if not self.connect():
            return False
        
        try:
            self._compute_embeddings()
            return True
        except Exception as e:
            logger.error(f"❌ Error loading graph embeddings: {e}")
            return False
    
    def _compute_embeddings(self):
        """Compute OpenAI embeddings for all nodes."""
        with self.driver.session() as session:
            # Get all nodes with their labels and names
            result = session.run("""
                MATCH (n)
                RETURN elementId(n) as node_id, labels(n)[0] as label, n.name as name
            """)
            
            # Collect all texts for batch processing
            node_data = []
            texts = []
            
            for record in result:
                node_id = record["node_id"]
                label = record["label"] or ""
                name = record["name"] or ""
                
                text = f"{label}: {name}"
                node_data.append((node_id, text))
                texts.append(text)
            
            if texts:
                try:
                    # Get embeddings in batches for efficiency
                    batch_size = 100
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        batch_node_data = node_data[i:i + batch_size]
                        
                        response = self.openai_client.embeddings.create(
                            input=batch_texts,
                            model=self.embedding_model
                        )
                        
                        for j, embedding_obj in enumerate(response.data):
                            node_id = batch_node_data[j][0]
                            embedding = np.array(embedding_obj.embedding)
                            self.node_embeddings[node_id] = embedding
                    
                    logger.info(f"✅ Computed OpenAI embeddings for {len(self.node_embeddings)} nodes")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to compute embeddings: {e}")
                    raise
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find nodes most similar to query using OpenAI embeddings."""
        if not self.node_embeddings:
            return []
        
        try:
            # Get query embedding from OpenAI
            response = self.openai_client.embeddings.create(
                input=[query],
                model=self.embedding_model
            )
            query_embedding = np.array(response.data[0].embedding)
            
            similarities = []
            
            for node_id, embedding in self.node_embeddings.items():
                sim = cosine_similarity([query_embedding], [embedding])[0][0]
                if sim >= 0.2:  # Reasonable threshold
                    similarities.append((node_id, sim))
            
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Error in semantic search: {e}")
            return []
    
    def expand_from_nodes(self, start_node_ids: List[str], max_nodes: int = 25, direct_only: bool = False) -> set:
        """Expand from starting nodes following meaningful connections (bi-directional, any type)."""
        if not start_node_ids:
            return set()
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start) WHERE elementId(start) IN $start_ids
                OPTIONAL MATCH path = (start)-[*1..2]-(n)
                WHERE n.name IS NOT NULL
                RETURN DISTINCT elementId(n) AS node_id
                LIMIT $max_nodes
            """, start_ids=start_node_ids, max_nodes=max_nodes)
            connected = {record["node_id"] for record in result}
            connected.update(start_node_ids)
            print(f"[DEBUG] expand_from_nodes: expanded node_ids = {connected}")
            return connected
    
    def query(self, query: str, direct_only: bool = False, max_nodes: int = 25) -> Dict:
        """Topic-centered query for GraphRAG: requires topic match, returns error if no match."""
        if not self.load_graph_with_embeddings():
            return {
                'query': query,
                'error': 'No Neo4j graph found. Process emails first.',
                'nodes': []
            }
        # Topic match required
        topic_matches = self.search_topics_by_name(query, semantic_threshold=0.5)
        if topic_matches:
            topic_ids = [topic_id for topic_id, score in topic_matches if score >= 0.5]
            all_node_ids = self.expand_from_nodes(
                topic_ids, max_nodes=max_nodes, direct_only=direct_only
            )
            all_nodes = self._get_node_names_from_ids(list(all_node_ids))
            confidence = topic_matches[0][1] if topic_matches else 0.0
            explanation = f"Found {len(all_nodes)} nodes from {len(topic_ids)} related topic(s)"
            if direct_only:
                explanation += " (direct neighbors only)"
            return {
                'query': query,
                'relevant_nodes': [(name, score) for name, score in self._convert_topic_matches_to_names(topic_matches) if score >= 0.5],
                'all_nodes': all_nodes,
                'confidence_score': round(confidence, 3),
                'explanation': explanation,
                'method': 'neo4j_topic_search'
            }
        # No topic match: return error and available topics
        available_topics = self._get_available_topics()
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

    def search_topics_by_name(self, query: str, semantic_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Search for topics using OpenAI embeddings with flexible matching."""
        if not self.driver:
            return []
        
        try:
            # Get query embedding from OpenAI
            response = self.openai_client.embeddings.create(
                input=[query],
                model=self.embedding_model
            )
            query_embedding = np.array(response.data[0].embedding)
            
            topic_matches = []
            
            with self.driver.session() as session:
                # Get all topic nodes
                result = session.run("""
                    MATCH (t:Topic)
                    RETURN elementId(t) as node_id, t.name as name
                """)
                
                for record in result:
                    node_id = record["node_id"]
                    topic_name = record["name"] or ""
                    
                    # Calculate semantic similarity
                    if node_id in self.node_embeddings:
                        topic_embedding = self.node_embeddings[node_id]
                        similarity = cosine_similarity([query_embedding], [topic_embedding])[0][0]
                        
                        # Also check for substring matches
                        topic_name_lower = topic_name.lower()
                        query_lower = query.lower()
                        
                        # Boost similarity for substring matches
                        if (query_lower in topic_name_lower or 
                            any(word in topic_name_lower for word in query_lower.split()) or
                            similarity >= semantic_threshold):
                            
                            # Give higher score to exact or close matches
                            if query_lower in topic_name_lower:
                                similarity = max(similarity, 0.9)
                            
                            topic_matches.append((node_id, similarity))
            
            return sorted(topic_matches, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error in topic search: {e}")
            return []
    
    def _get_node_names_from_ids(self, node_ids: List[str]) -> List[str]:
        """Convert internal node IDs to node names."""
        if not node_ids:
            return []
        
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $node_ids as node_id
                MATCH (n) WHERE elementId(n) = node_id
                RETURN n.name as name
            """, node_ids=node_ids)
            
            return [record["name"] for record in result if record["name"]]
    
    def _convert_topic_matches_to_names(self, topic_matches: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Convert topic matches from IDs to names."""
        if not topic_matches:
            return []
        
        node_ids = [node_id for node_id, score in topic_matches]
        
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $node_ids as node_id
                MATCH (n) WHERE elementId(n) = node_id
                RETURN elementId(n) as node_id, n.name as name
            """, node_ids=node_ids)
            
            id_to_name = {record["node_id"]: record["name"] for record in result}
            
            return [(id_to_name.get(node_id, "Unknown"), score) 
                    for node_id, score in topic_matches if node_id in id_to_name]
    
    def _get_available_topics(self) -> List[str]:
        """Get list of all available topic names."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.name as name
                ORDER BY t.name
            """)
            
            return [record["name"] for record in result if record["name"]]
    
    def generate_visualization_html(self, query: str, result: Dict) -> str:
        """Generate visualization HTML for Neo4j graph data."""
        try:
            from pyvis.network import Network
        except ImportError:
            return "<p>pyvis not installed</p>"
        
        if not self.driver:
            return "<p>No Neo4j connection</p>"
        
        try:
            net = Network(height="600px", width="100%")
            
            # Get nodes and relationships for the result
            nodes_to_show = result.get('all_nodes', [])
            if not nodes_to_show:
                return "<p>No nodes found in query result</p>"
            
            with self.driver.session() as session:
                # Get nodes with their properties
                node_result = session.run("""
                    UNWIND $node_names as name
                    MATCH (n {name: name})
                    RETURN n.name as name, labels(n)[0] as label, properties(n) as props
                """, node_names=nodes_to_show)
                
                # Colors for different node types
                colors = {
                    'Topic': '#FF6B9D',      # Pink - most important
                    'Task': '#90EE90',       # Light green
                    'Person': '#87CEEB',     # Sky blue  
                    'Role': '#FFA500',       # Orange
                    'Department': '#DDA0DD', # Plum
                    'Organization': '#F0E68C', # Khaki
                    'Date': '#D3D3D3',       # Light gray
                    'Summary': '#FFE4B5',    # Moccasin
                    'EmailIndex': '#E6E6FA'  # Lavender
                }
                
                # Add nodes
                for record in node_result:
                    name = record["name"]
                    label = record["label"]
                    props = record["props"]
                    
                    color = colors.get(label, '#BDC3C7')
                    
                    # Node sizing
                    if label == 'Topic':
                        node_size = 25
                    elif label == 'Task':
                        node_size = 20
                    elif label == 'Person':
                        node_size = 15
                    else:
                        node_size = 12
                    
                    # Display name
                    if label == 'Task':
                        display_name = name[:25] + "..." if len(name) > 25 else name
                    elif label == 'Summary':
                        display_name = name[:30] + "..." if len(name) > 30 else name
                    else:
                        display_name = name[:20] + "..." if len(name) > 20 else name
                    
                    # Tooltip
                    tooltip_parts = [f"<b>{label}</b>: {name}"]
                    for key, value in props.items():
                        if key != 'name' and value:
                            tooltip_parts.append(f"{key}: {value}")
                    
                    tooltip = "<br>".join(tooltip_parts)
                    
                    net.add_node(
                        name,
                        label=display_name,
                        title=tooltip,
                        color=color,
                        size=node_size,
                        font={'size': 10, 'color': 'black'}
                    )
                
                # Get relationships between these nodes
                rel_result = session.run("""
                    UNWIND $node_names as name1
                    MATCH (n1 {name: name1})
                    MATCH (n1)-[r]->(n2)
                    WHERE n2.name IN $node_names
                    RETURN n1.name as from_node, type(r) as rel_type, n2.name as to_node
                """, node_names=nodes_to_show)
                
                # Add edges
                for record in rel_result:
                    net.add_edge(
                        record["from_node"], 
                        record["to_node"], 
                        label=record["rel_type"]
                    )
            
            # Set heading and generate HTML
            net.heading = f"Neo4j Query: {query}"
            html_content = net.generate_html()
            return html_content
            
        except Exception as e:
            return f"<p>Error generating Neo4j visualization: {str(e)}</p>"
        finally:
            self.close()

    def query_flexible(self, query: str, max_nodes: int = 25) -> Dict:
        """
        Entity-agnostic query: match any node type by name, expand subgraph, return context for answer/visualization.
        """
        if not self.load_graph_with_embeddings():
            return {
                'query': query,
                'error': 'No Neo4j graph found. Process emails first.',
                'nodes': []
            }
        # 1. Semantic search across all nodes
        response = self.openai_client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        query_embedding = np.array(response.data[0].embedding)
        similarities = []
        node_id_to_name = {}
        node_id_to_label = {}
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN elementId(n) as node_id, labels(n)[0] as label, n.name as name
            """)
            for record in result:
                node_id = record["node_id"]
                label = record["label"]
                name = record["name"]
                node_id_to_name[node_id] = name
                node_id_to_label[node_id] = label
                if node_id in self.node_embeddings:
                    node_embedding = self.node_embeddings[node_id]
                    sim = cosine_similarity([query_embedding], [node_embedding])[0][0]
                    similarities.append((node_id, sim))
        # Sort by similarity
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        # Use top match(es) above a threshold
        matched_nodes = [node_id for node_id, sim in similarities if sim >= 0.5]
        if not matched_nodes and similarities:
            # Fallback: take the best match
            matched_nodes = [similarities[0][0]]
        if not matched_nodes:
            return {
                'query': query,
                'error': 'No matching node found in graph.',
                'nodes': []
            }
        # 2. Expand from matched node(s) (undirected, multi-hop)
        all_node_ids = self.expand_from_nodes(matched_nodes, max_nodes=max_nodes, direct_only=False)
        all_nodes = self._get_node_names_from_ids(list(all_node_ids))
        matched_names = [node_id_to_name[nid] for nid in matched_nodes if nid in node_id_to_name]
        matched_labels = [node_id_to_label[nid] for nid in matched_nodes if nid in node_id_to_label]
        explanation = f"Expanded from {matched_names} ({matched_labels}), found {len(all_nodes)} nodes."
        return {
            'query': query,
            'matched_nodes': matched_names,
            'matched_labels': matched_labels,
            'all_nodes': all_nodes,
            'confidence_score': similarities[0][1] if similarities else 0.0,
            'explanation': explanation,
            'method': 'neo4j_flexible_search'
        }
    # Compatibility methods
    def query_with_semantic_reasoning(self, query: str) -> Dict:
        """Enhanced query with RAGAS evaluation for comprehensive assessment."""
        # Use entity-agnostic flexible query for semantic reasoning
        result = self.query_flexible(query)
        print("[DEBUG] Matched nodes:", result.get("matched_nodes"))
        print("[DEBUG] Expanded all_nodes:", result.get("all_nodes"))

        # If there's an error, return early
        if 'error' in result:
            return result

        # Try to get visualization and RAGAS evaluation if we have enough context
        if result.get('all_nodes'):
            try:
                # Visualize graph
                result["graph_html"] = self.generate_visualization_html(query, result)

                # Import RAGAS evaluator
                from utils.ragas_evaluator import RAGASEvaluator

                # Extract contexts from the graph result
                contexts = self._extract_contexts_from_nodes(result.get('all_nodes', []))

                # Get the generated response from format_response
                response = format_response(result)

                # Initialize RAGAS evaluator and get scores
                evaluator = RAGASEvaluator()
                import asyncio

                # Handle async evaluation
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If in async context, create task for later
                        ragas_scores = {}
                        logger.info("Async context detected - RAGAS evaluation skipped in real-time")
                    else:
                        # Always pass ground_truth as the context for reference-free RAGAS
                        ground_truth = contexts[0] if contexts else ""
                        ragas_scores = asyncio.run(
                            evaluator.evaluate_single(query, response, contexts, ground_truth=ground_truth)
                        )
                except:
                    # Fallback to simple confidence if RAGAS fails
                    ragas_scores = {}
                    logger.debug("RAGAS evaluation fallback - using confidence score")

                # Add RAGAS scores to result
                if ragas_scores:
                    result['ragas_scores'] = ragas_scores
                    result['ragas_overall'] = evaluator.get_overall_score(ragas_scores)
                    result['ragas_summary'] = evaluator.format_evaluation_summary(ragas_scores)

                    # Use RAGAS overall score as primary confidence if available
                    result['confidence_score'] = result['ragas_overall']
                else:
                    # Keep original confidence score
                    result['ragas_scores'] = {}
                    result['ragas_overall'] = result.get('confidence_score', 0.0)
                    result['ragas_summary'] = '⚠️ RAGAS evaluation unavailable. No RAGAS scores could be computed.'

            except Exception as e:
                logger.debug(f"RAGAS evaluation unavailable: {e}")
                # Fallback to original confidence
                result['ragas_scores'] = {}
                result['ragas_overall'] = result.get('confidence_score', 0.0)
                result['ragas_summary'] = f"📊 **Quick Assessment:** {result.get('confidence_score', 0.0):.3f}"

        return result
    
    def _extract_contexts_from_nodes(self, node_names: List[str]) -> List[str]:
        """Extract meaningful context strings from node names for RAGAS evaluation."""
        contexts = []
        
        with self.driver.session() as session:
            # Get detailed information for each node
            for node_name in node_names[:10]:  # Limit to 10 nodes for context
                try:
                    result = session.run("""
                        MATCH (n {name: $name})
                        RETURN labels(n)[0] as label, properties(n) as props
                    """, name=node_name)
                    
                    for record in result:
                        label = record["label"]
                        props = record["props"]
                        
                        # Create meaningful context based on node type
                        if label == "Task":
                            context = f"Task: {props.get('name', '')}. Summary: {props.get('summary', 'No summary available')}."
                        elif label == "Person":
                            context = f"Person: {props.get('name', '')} with role {props.get('role', 'Unknown role')}."
                        elif label == "Topic":
                            context = f"Project Topic: {props.get('name', '')}."
                        elif label == "Summary":
                            context = f"Summary: {props.get('name', '')}"
                        else:
                            context = f"{label}: {props.get('name', '')}"
                        
                        if context.strip():
                            contexts.append(context)
                            
                except Exception as e:
                    logger.debug(f"Error extracting context for {node_name}: {e}")
                    continue
        
        # Ensure we have some context
        if not contexts:
            contexts = [f"Graph contains information about: {', '.join(node_names[:5])}"]
        print("[DEBUG] Final contexts sent to RAGAS:", contexts)
        return contexts

    def _get_all_node_names(self, limit=100):
        """Return names of all nodes in the graph, up to a limit."""
        if not self.driver:
            return []
        names = []
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN n.name AS name LIMIT $limit", limit=limit)
            for record in result:
                if record["name"]:
                    names.append(record["name"])
        return names


def format_response(result: Dict) -> str:
    """Format response for Neo4j GraphRAG results."""
    if 'error' in result:
        return result['error']
    
    if not result.get('all_nodes'):
        return "No information found."
    
    try:
        # Connect to Neo4j to get task details
        graphrag = Neo4jGraphRAG()
        if not graphrag.connect():
            return "Error connecting to Neo4j database."
        
        with graphrag.driver.session() as session:
            # Find all tasks in the result
            task_result = session.run("""
                UNWIND $node_names as name
                MATCH (task:Task {name: name})
                RETURN task.name as task_name
            """, node_names=result.get('all_nodes', []))
            
            task_names = [record["task_name"] for record in task_result]
            
            if not task_names:
                return "No tasks found in the results."
            
            # Format each task
            response_parts = []
            
            for task_name in task_names:
                task_info = [f"**Task:** {task_name}"]
                
                # Get task details and relationships
                detail_result = session.run("""
                    MATCH (task:Task {name: $task_name})
                    
                    // Get topic
                    OPTIONAL MATCH (topic:Topic)-[:HAS_TASK]->(task)
                    
                    // Get dates
                    OPTIONAL MATCH (task)-[:SENT_ON]->(sent_date:Date)
                    OPTIONAL MATCH (task)-[:DUE_ON]->(due_date:Date)
                    
                    // Get summary
                    OPTIONAL MATCH (task)-[:BASED_ON]->(summary:Summary)
                    
                    // Get email
                    OPTIONAL MATCH (task)-[:LINKED_TO]->(email:EmailIndex)
                    
                    // Get owner
                    OPTIONAL MATCH (task)-[:ASSIGNED_TO]->(owner:Person)
                    
                    // Get collaborators
                    OPTIONAL MATCH (task)-[:COLLABORATED_ON]->(collab:Person)
                    
                    RETURN topic.name as topic_name,
                           sent_date.name as sent_date,
                           due_date.name as due_date,
                           summary.name as summary_text,
                           email.name as email_index,
                           owner.name as owner_name,
                           collect(DISTINCT collab.name) as collaborators
                """, task_name=task_name)
                
                for record in detail_result:
                    if record["topic_name"]:
                        task_info.append(f"**Topic:** {record['topic_name']}")
                    if record["sent_date"]:
                        task_info.append(f"   • **Sent Date:** {record['sent_date']}")
                    if record["due_date"]:
                        task_info.append(f"   • **Due Date:** {record['due_date']}")
                    if record["summary_text"]:
                        task_info.append(f"   • **Summary:** {record['summary_text']}")
                    if record["email_index"]:
                        task_info.append(f"   • **Email Index:** {record['email_index']}")
                    if record["owner_name"]:
                        task_info.append(f"   • **Responsible To:** {record['owner_name']}")
                    
                    collaborators = [c for c in record["collaborators"] if c]
                    if collaborators:
                        task_info.append(f"   • **Collaborated By:** {', '.join(collaborators)}")
                
                response_parts.append("\n".join(task_info))
        
        graphrag.close()
        
        # Add confidence
        confidence = result.get('confidence_score', 0.0)
        conf_text = "🟢 High" if confidence > 0.7 else "🟡 Medium" if confidence > 0.4 else "🔴 Low"
        response_parts.append(f"\n**Confidence:** {conf_text} ({confidence})")
        
        return "\n\n".join(response_parts)
        
    except Exception as e:
        return f"📊 Error formatting Neo4j response: {str(e)}"


# Compatibility function
def format_graphrag_response(result: Dict) -> str:
    """Compatibility function."""
    return format_response(result)



