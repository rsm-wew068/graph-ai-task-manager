import pickle
import difflib
from pyvis.network import Network
import os
import networkx as nx

# Basic resolver (exact and fuzzy match fallback)
def resolve_topic_node(G, name_guess):
    topic_nodes = [n for n, d in G.nodes(data=True) if d.get("label") == "Topic"]

    # Exact match
    for node in topic_nodes:
        if node.lower() == name_guess.lower():
            return node

    # Fuzzy match
    matches = difflib.get_close_matches(name_guess.lower(), [n.lower() for n in topic_nodes], n=1, cutoff=0.7)
    if matches:
        return next(n for n in topic_nodes if n.lower() == matches[0])

    return None

# Utility function to infer topic name from query
def infer_topic_name(query: str) -> str:
    return query.split()[-1]  # placeholder fallback if no GPT used here

def get_topic_subgraph(topic_name: str, graph_path="topic_graph.gpickle"):
    try:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        return None, f"Failed to load graph: {e}"

    topic = resolve_topic_node(G, topic_name)
    if not topic:
        return None, f"No topic found for '{topic_name}'"

    tasks = []
    for task in G.neighbors(topic):
        if G[topic][task].get("label") != "HAS_TASK":
            continue

        task_info = {
            "task": task,
            "start": None,
            "due": None,
            "summary": None,
            "email_index": None,
            "responsible": [],
            "collaborators": [],
        }

        for neighbor in G.neighbors(task):
            label = G[task][neighbor].get("label")
            if label == "START_ON":
                task_info["start"] = neighbor
            elif label == "DUE_ON":
                task_info["due"] = neighbor
            elif label == "BASED_ON":
                task_info["summary"] = G.nodes[neighbor].get("name", neighbor)
            elif label == "LINKED_TO":
                task_info["email_index"] = neighbor
            elif label in {"RESPONSIBLE_TO", "COLLABORATED_BY"}:
                person_info = {"name": neighbor, "relation": label, "role": None, "department": None, "organization": None}
                for role in G.neighbors(neighbor):
                    if G[neighbor][role].get("label") == "HAS_ROLE":
                        person_info["role"] = role
                        for dept in G.neighbors(role):
                            if G[role][dept].get("label") == "BELONGS_TO":
                                person_info["department"] = dept
                                for org in G.neighbors(dept):
                                    if G[dept][org].get("label") == "IS_IN":
                                        person_info["organization"] = org
                                        break
                                break
                        break
                if label == "RESPONSIBLE_TO":
                    task_info["responsible"].append(person_info)
                else:
                    task_info["collaborators"].append(person_info)

        tasks.append(task_info)

    return tasks, None

def format_graph_observation(graph_data):
    if not graph_data:
        return "Observation: No data found."

    response = []
    for task in graph_data:
        task_lines = [f"Task: {task['task']}"]
        if task['start']: task_lines.append(f"  • Start Date: {task['start']}")
        if task['due']: task_lines.append(f"  • Due Date: {task['due']}")
        if task['summary']: task_lines.append(f"  • Summary: {task['summary']}")
        if task['email_index']: task_lines.append(f"  • Email Index: {task['email_index']}")

        for role_group in [("responsible", "Responsible To"), ("collaborators", "Collaborated By")]:
            for person in task[role_group[0]]:
                parts = [f"{role_group[1]}: {person['name']}"]
                if person["role"]: parts.append(f"Role: {person['role']}")
                if person["department"]: parts.append(f"Department: {person['department']}")
                if person["organization"]: parts.append(f"Organization: {person['organization']}")
                task_lines.append("  • " + " | ".join(parts))

        response.append("\n".join(task_lines))
    return "\n\n".join(response)

def build_and_save_graph(topic, graph_path="topic_graph.gpickle"):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    actual_node = resolve_topic_node(G, topic)
    if not actual_node:
        print(f"❌ Could not resolve topic name: {topic}")
        return None
    ego = nx.ego_graph(G, actual_node, radius=6)
    net = Network(height="500px", width="100%")
    for node, data in ego.nodes(data=True):
        node_color = "pink" if node == actual_node else "lightblue"
        display_label = data.get("name") or data.get("label") or str(node)
        net.add_node(
            node,
            label=display_label,
            title="<br>".join([f"{k}: {v}" for k, v in data.items() if k != "label"]),
            color=node_color,
        )
    for u, v, d in ego.edges(data=True):
        net.add_edge(u, v, label=d.get("label", ""))
    os.makedirs("static", exist_ok=True)
    filename = f"static/graph_{actual_node.replace(' ', '_')}.html"
    net.save_graph(filename)
    return filename

def query_graph(topic_info, graph_path="topic_graph.gpickle"):
    """Query the graph for information about a topic."""
    try:
        if not os.path.exists(graph_path):
            return "No topic graph found. Please process some emails first."
        
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        
        topic_name = topic_info.get("name", "")
        actual_node = resolve_topic_node(G, topic_name)
        
        if not actual_node:
            return f"Could not find topic: {topic_name}"
        
        # Get subgraph around the topic
        ego = nx.ego_graph(G, actual_node, radius=3)
        
        # Format the graph data as text
        return format_graph_observation(ego)
    
    except Exception as e:
        return f"Error querying graph: {str(e)}"

def visualize_graph(topic_name, graph_path="topic_graph.gpickle"):
    """Create a visualization of the topic graph."""
    try:
        return build_and_save_graph(topic_name, graph_path)
    except Exception as e:
        print(f"Error visualizing graph: {str(e)}")
        return None