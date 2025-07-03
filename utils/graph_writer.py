import networkx as nx
import pickle

def write_tasks_to_graph(data, save_path=None):
    """
    Builds a directed NetworkX graph from a list of topic dictionaries.
    Optionally saves the graph to disk if save_path is provided.
    Returns the NetworkX graph object.
    """
    G = nx.DiGraph()

    for entry in data:
        topic = entry["Topic"]
        pname = topic["name"]
        G.add_node(pname, label="Topic", name=pname)

        for t in topic.get("tasks", []):
            task = t["task"]
            tname = task["name"]
            start = task["start_date"]
            due = task["due_date"]
            email_index = t.get("email_index")
            summary_text = task.get("summary", "")

            G.add_node(tname, label="Task", name=tname)
            G.add_edge(pname, tname, label="HAS_TASK")

            if start:
                G.add_node(start, label="Date", name=start)
                G.add_edge(tname, start, label="START_ON")
            if due:
                G.add_node(due, label="Date", name=due)
                G.add_edge(tname, due, label="DUE_ON")

            if summary_text:
                summary_node = f"Summary: {summary_text}"
                G.add_node(summary_node, label="Summary", name=summary_text)
                G.add_edge(tname, summary_node, label="BASED_ON")

            if email_index:
                G.add_node(email_index, label="Email Index", name=email_index)
                G.add_edge(tname, email_index, label="LINKED_TO")

            org = t["owner"].get("organization", "Unknown Org")
            G.add_node(org, label="Organization", name=org)

            owner = t["owner"]
            G.add_node(owner["name"], label="Person", name=owner["name"])
            role_name = f"{owner['role']} ({owner['department']})"
            G.add_node(role_name, label="Role", name=role_name)
            G.add_node(owner["department"], label="Department", name=owner["department"])

            G.add_edge(tname, owner["name"], label="RESPONSIBLE_TO")
            G.add_edge(owner["name"], role_name, label="HAS_ROLE")
            G.add_edge(role_name, owner["department"], label="BELONGS_TO")
            G.add_edge(owner["department"], org, label="IS_IN")

            for c in task.get("collaborators", []):
                role_name = f"{c['role']} ({c['department']})"
                G.add_node(c["name"], label="Person", name=c["name"])
                G.add_node(role_name, label="Role", name=role_name)
                G.add_node(c["department"], label="Department", name=c["department"])
                G.add_edge(tname, c["name"], label="COLLABORATED_BY")
                G.add_edge(c["name"], role_name, label="HAS_ROLE")
                G.add_edge(role_name, c["department"], label="BELONGS_TO")
                G.add_edge(c["department"], org, label="IS_IN")

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(G, f)
    return G