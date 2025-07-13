from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def write_tasks_to_neo4j(data):
    """
    Writes tasks and relationships to Neo4j using the clean schema:
    • (:Topic)-[:HAS_TASK]->(:Task)
    • (:Task)-[:START_ON]->(:Date)
    • (:Task)-[:DUE_ON]->(:Date)
    • (:Task)-[:LINKED_TO]->(:Email_Index)
    • (:Task)-[:BASED_ON]->(:Summary)
    • (:Task)-[:RESPONSIBLE_TO]->(:Person)
    • (:Task)-[:COLLABORATED_BY]->(:Person)
    • (:Person)-[:HAS_ROLE]->(:Role)
    • (:Role)-[:BELONGS_TO]->(:Department)
    • (:Department)-[:IS_IN]->(:Organization)
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for entry in data:
            topic = entry["Topic"]
            topic_name = topic["name"]
            session.write_transaction(_merge_topic, topic_name)
            
            for t in topic.get("tasks", []):
                task = t["task"]
                task_name = task["name"]
                start_date = task.get("start_date", "")
                due_date = task.get("due_date", "")
                email_index = t.get("email_index", "")
                summary_text = task.get("summary", "")
                
                # Create task and its basic relationships
                session.write_transaction(_merge_task, topic_name, task_name, start_date, due_date, email_index, summary_text)
                
                # Handle owner (person responsible for the task)
                owner = task.get("owner", {})
                if owner.get("name") and owner.get("name") != "Unknown":
                    session.write_transaction(_merge_owner, task_name, owner)
                
                # Handle collaborators
                for collaborator in task.get("collaborators", []):
                    if collaborator.get("name") and collaborator.get("name") != "Unknown":
                        session.write_transaction(_merge_collaborator, task_name, collaborator)
    
    driver.close()

def _merge_topic(tx, topic_name):
    """Create or merge a topic node."""
    tx.run("""
        MERGE (t:Topic {name: $topic_name})
        SET t.node_type = 'Topic'
    """, topic_name=topic_name)

def _merge_task(tx, topic_name, task_name, start_date, due_date, email_index, summary_text):
    """Create task and its relationships to dates, email, and summary."""
    tx.run("""
        MERGE (task:Task {name: $task_name})
        SET task.start_date = $start_date,
            task.due_date = $due_date,
            task.email_id = $email_index,
            task.summary = $summary_text,
            task.topic = $topic_name,
            task.node_type = 'Task'
        WITH task
        MATCH (topic:Topic {name: $topic_name})
        MERGE (topic)-[:HAS_TASK]->(task)
        
        // Start date relationship
        FOREACH (sd IN CASE WHEN $start_date IS NOT NULL AND $start_date <> '' THEN [1] ELSE [] END |
            MERGE (start_date:Date {name: $start_date})
            SET start_date.node_type = 'Date',
                start_date.date_type = 'start'
            MERGE (task)-[:START_ON]->(start_date)
        )
        
        // Due date relationship
        FOREACH (dd IN CASE WHEN $due_date IS NOT NULL AND $due_date <> '' THEN [1] ELSE [] END |
            MERGE (due_date:Date {name: $due_date})
            SET due_date.node_type = 'Date',
                due_date.date_type = 'due'
            MERGE (task)-[:DUE_ON]->(due_date)
        )
        
        // Email index relationship
        FOREACH (em IN CASE WHEN $email_index IS NOT NULL AND $email_index <> '' THEN [1] ELSE [] END |
            MERGE (email:Email_Index {name: $email_index})
            SET email.node_type = 'Email'
            MERGE (task)-[:LINKED_TO]->(email)
        )
        
        // Summary relationship
        FOREACH (sm IN CASE WHEN $summary_text IS NOT NULL AND $summary_text <> '' THEN [1] ELSE [] END |
            MERGE (summary:Summary {name: $summary_text})
            SET summary.node_type = 'Summary'
            MERGE (task)-[:BASED_ON]->(summary)
        )
    """, topic_name=topic_name, task_name=task_name, start_date=start_date, 
         due_date=due_date, email_index=email_index, summary_text=summary_text)

def _merge_owner(tx, task_name, person_data):
    """
    Create the full person hierarchy and connect as OWNER of the task.
    """
    person_name = person_data.get("name", "Unknown")
    person_role = person_data.get("role", "Unknown")
    person_dept = person_data.get("department", "Unknown")
    person_org = person_data.get("organization", "Unknown")
    
    # Skip if person name is not meaningful
    if person_name in ["Unknown", "", None]:
        return
    
    # Handle null/empty values for organization and department
    if not person_org or person_org in ["Unknown", "", None]:
        person_org = "Unknown Organization"
    if not person_dept or person_dept in ["Unknown", "", None]:
        person_dept = "Unknown Department"
    if not person_role or person_role in ["Unknown", "", None]:
        person_role = "Unknown Role"
    
    tx.run("""
        // Create organization hierarchy
        MERGE (org:Organization {name: $org})
        SET org.node_type = 'Organization'
        MERGE (dept:Department {name: $dept})-[:IS_IN]->(org)
        SET dept.node_type = 'Department',
            dept.organization = $org
        MERGE (role:Role {name: $role})-[:BELONGS_TO]->(dept)
        SET role.node_type = 'Role',
            role.department = $dept,
            role.organization = $org
        
        // Create person and connect to role
        MERGE (person:Person {name: $person_name})-[:HAS_ROLE]->(role)
        SET person.node_type = 'Person',
            person.role = $role,
            person.department = $dept,
            person.organization = $org,
            person.relationship_type = 'owner'
        
        // Connect person as OWNER of the task
        WITH person
        MATCH (task:Task {name: $task_name})
        MERGE (task)-[:RESPONSIBLE_TO]->(person)
    """, org=person_org, dept=person_dept, role=person_role, 
         person_name=person_name, task_name=task_name)

def _merge_collaborator(tx, task_name, person_data):
    """
    Create the full person hierarchy and connect as COLLABORATOR of the task.
    """
    person_name = person_data.get("name", "Unknown")
    person_role = person_data.get("role", "Unknown")
    person_dept = person_data.get("department", "Unknown")
    person_org = person_data.get("organization", "Unknown")
    
    # Skip if person name is not meaningful
    if person_name in ["Unknown", "", None]:
        return
    
    # Handle null/empty values for organization and department
    if not person_org or person_org in ["Unknown", "", None]:
        person_org = "Unknown Organization"
    if not person_dept or person_dept in ["Unknown", "", None]:
        person_dept = "Unknown Department"
    if not person_role or person_role in ["Unknown", "", None]:
        person_role = "Unknown Role"
    
    tx.run("""
        // Create organization hierarchy
        MERGE (org:Organization {name: $org})
        SET org.node_type = 'Organization'
        MERGE (dept:Department {name: $dept})-[:IS_IN]->(org)
        SET dept.node_type = 'Department',
            dept.organization = $org
        MERGE (role:Role {name: $role})-[:BELONGS_TO]->(dept)
        SET role.node_type = 'Role',
            role.department = $dept,
            role.organization = $org
        
        // Create person and connect to role
        MERGE (person:Person {name: $person_name})-[:HAS_ROLE]->(role)
        SET person.node_type = 'Person',
            person.role = $role,
            person.department = $dept,
            person.organization = $org,
            person.relationship_type = 'collaborator'
        
        // Connect person as COLLABORATOR of the task
        WITH person
        MATCH (task:Task {name: $task_name})
        MERGE (task)-[:COLLABORATED_BY]->(person)
    """, org=person_org, dept=person_dept, role=person_role, 
         person_name=person_name, task_name=task_name)
