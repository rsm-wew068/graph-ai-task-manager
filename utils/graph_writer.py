import networkx as nx
import pickle


def write_tasks_to_graph(data, save_path=None):
    """
    Builds a directed NetworkX graph from a list of topic dictionaries.
    Follows the legacy build_graph.py approach exactly, but using "Topic"
    instead of "Project".
    Returns the NetworkX graph object.
    """
    G = nx.DiGraph()

    for entry in data:
        topic = entry["Topic"]
        topic_name = topic["name"]

        # Add topic node (equivalent to Project in legacy)
        G.add_node(topic_name, label="Topic", name=topic_name)

        for t in topic.get("tasks", []):
            task = t["task"]
            task_name = task["name"]
            start_date = task["start_date"]
            due_date = task["due_date"]
            email_index = t.get("email_index")
            summary_text = task.get("summary", "")

            # Add task node and connect to topic
            G.add_node(task_name, label="Task", name=task_name)
            G.add_edge(topic_name, task_name, label="HAS_TASK")

            # Add date nodes and connections (exactly as in legacy)
            if start_date:
                G.add_node(start_date, label="Date", name=start_date)
                G.add_edge(task_name, start_date, label="START_ON")
            if due_date:
                G.add_node(due_date, label="Date", name=due_date)
                G.add_edge(task_name, due_date, label="DUE_ON")

            # Add summary node and connection (exactly as in legacy)
            if summary_text:
                summary_node = f"Summary: {summary_text}"
                G.add_node(summary_node, label="Summary", name=summary_text)
                G.add_edge(task_name, summary_node, label="BASED_ON")

            # Add email index node and connection (exactly as in legacy)
            if email_index:
                G.add_node(email_index, label="Email Index", name=email_index)
                G.add_edge(task_name, email_index, label="LINKED_TO")

            # Handle owner - no fallback defaults, Unknown is Unknown
            owner = task.get("owner", {})
            if isinstance(owner, dict) and owner:
                # Get organization (no fallback, Unknown if not specified)
                org = owner.get("organization", "Unknown")
                G.add_node(org, label="Organization", name=org)

                # Add owner person node
                owner_name = owner.get("name", "Unknown")
                owner_role = owner.get("role", "Unknown")
                person_name = f"{owner_name} ({owner_role})"
                G.add_node(person_name, label="Person", name=person_name)
                
                # Add owner role node (following legacy format exactly)
                owner_role = owner.get("role", "Unknown")
                owner_dept = owner.get("department", "Unknown")
                role_name = f"{owner_role} ({owner_dept})"
                dept_name = f"{owner_dept} ({org})"  # dept (org) format
                G.add_node(role_name, label="Role", name=role_name)
                G.add_node(dept_name, label="Department", name=dept_name)

                # Create owner relationship chain (exactly as in legacy)
                G.add_edge(task_name, person_name, label="RESPONSIBLE_TO")
                G.add_edge(person_name, role_name, label="HAS_ROLE")
                G.add_edge(role_name, dept_name, label="BELONGS_TO")
                G.add_edge(dept_name, org, label="IS_IN")
            else:
                # Fallback for missing or invalid owner - all Unknown
                org = "Unknown"
                G.add_node(org, label="Organization", name=org)
                person_name = "Unknown (Unknown)"
                G.add_node(person_name, label="Person", name=person_name)
                role_name = "Unknown (Unknown)"
                G.add_node(role_name, label="Role", name=role_name)
                dept_name = "Unknown (Unknown)"  # dept (org) format
                G.add_node(dept_name, label="Department", name=dept_name)
                
                G.add_edge(task_name, person_name, label="RESPONSIBLE_TO")
                G.add_edge(person_name, role_name, label="HAS_ROLE")
                G.add_edge(role_name, dept_name, label="BELONGS_TO")
                G.add_edge(dept_name, org, label="IS_IN")

            # Handle collaborators (following legacy approach exactly)
            for c in task.get("collaborators", []):
                if isinstance(c, dict) and c:
                    collab_name = c.get("name", "Unknown")
                    collab_role = c.get("role", "Unknown")
                    collab_dept = c.get("department", "Unknown")
                    
                    # Create collaborator nodes with consistent formatting
                    person_name = f"{collab_name} ({collab_role})"
                    role_name = f"{collab_role} ({collab_dept})"
                    dept_name = f"{collab_dept} ({org})"  # dept (org)
                    
                    # Add collaborator nodes
                    G.add_node(person_name, label="Person", name=person_name)
                    G.add_node(role_name, label="Role", name=role_name)
                    G.add_node(dept_name, label="Department", name=dept_name)
                    
                    # Create collaborator relationship chain (as in legacy)
                    G.add_edge(task_name, person_name, label="COLLABORATED_BY")
                    G.add_edge(person_name, role_name, label="HAS_ROLE")
                    G.add_edge(role_name, dept_name, label="BELONGS_TO")
                    G.add_edge(dept_name, org, label="IS_IN")

    # Save graph if path provided
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(G, f)
    
    return G
