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
            # Handle task field safely with defaults
            task = t.get("task", {})
            if isinstance(task, dict):
                tname = task.get("name", "Unnamed Task")
                start = task.get("start_date", "")
                due = task.get("due_date", "")
                summary_text = task.get("summary", "")
            else:
                # If task is not a dict, create defaults
                tname = "Unnamed Task"
                start = ""
                due = ""
                summary_text = ""

            email_index = t.get("email_index")

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

            # Handle owner field safely with defaults
            owner = t.get("owner", {})
            if isinstance(owner, dict):
                org = owner.get("organization", "Unknown Org")
                owner_name = owner.get("name", "Unknown Owner")
                owner_role = owner.get("role", "Unknown Role")
                owner_dept = owner.get("department", "Unknown Department")
            elif isinstance(owner, str):
                # If owner is just a string, use it as the name
                org = "Unknown Org"
                owner_name = owner
                owner_role = "Unknown Role"
                owner_dept = "Unknown Department"
            else:
                # Fallback for any other type
                org = "Unknown Org"
                owner_name = "Unknown Owner"
                owner_role = "Unknown Role"
                owner_dept = "Unknown Department"

            G.add_node(org, label="Organization", name=org)
            G.add_node(owner_name, label="Person", name=owner_name)
            role_name = f"{owner_role} ({owner_dept})"
            G.add_node(role_name, label="Role", name=role_name)
            G.add_node(owner_dept, label="Department", name=owner_dept)

            G.add_edge(tname, owner_name, label="RESPONSIBLE_TO")
            G.add_edge(owner_name, role_name, label="HAS_ROLE")
            G.add_edge(role_name, owner_dept, label="BELONGS_TO")
            G.add_edge(owner_dept, org, label="IS_IN")

            # Handle collaborators safely
            for c in task.get("collaborators", []):
                if isinstance(c, dict):
                    c_name = c.get("name", "Unknown Collaborator")
                    c_role = c.get("role", "Unknown Role")
                    c_dept = c.get("department", "Unknown Department")
                else:
                    c_name = "Unknown Collaborator"
                    c_role = "Unknown Role"
                    c_dept = "Unknown Department"

                collab_role_name = f"{c_role} ({c_dept})"
                G.add_node(c_name, label="Person", name=c_name)
                G.add_node(collab_role_name, label="Role", name=collab_role_name)
                G.add_node(c_dept, label="Department", name=c_dept)
                G.add_edge(tname, c_name, label="COLLABORATED_BY")
                G.add_edge(c_name, collab_role_name, label="HAS_ROLE")
                G.add_edge(collab_role_name, c_dept, label="BELONGS_TO")
                G.add_edge(c_dept, org, label="IS_IN")

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(G, f)
    return G
