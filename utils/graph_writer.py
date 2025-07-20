from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def write_tasks_to_neo4j(tasks):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for task in tasks:
            session.write_transaction(_merge_task_and_relations, task)
    driver.close()

def _merge_task_and_relations(tx, task):
    tx.run("""
        MERGE (t:Task {email_source: $email_source})
        SET t.name = $name,
            t.node_type = 'Task'

        // Topic
        FOREACH (_ IN CASE WHEN $topic IS NOT NULL THEN [1] ELSE [] END |
            MERGE (topic:Topic {name: $topic})
            SET topic.node_type = 'Topic'
            MERGE (t)-[:HAS_TOPIC]->(topic)
        )

        // Received Date
        FOREACH (_ IN CASE WHEN $received_date IS NOT NULL THEN [1] ELSE [] END |
            MERGE (recv:ReceivedDate {value: $received_date})
            SET recv.node_type = 'ReceivedDate'
            MERGE (t)-[:RECEIVED_ON]->(recv)
        )

        // Due Date and related Status and Priority
        FOREACH (_ IN CASE WHEN $due_date IS NOT NULL THEN [1] ELSE [] END |
            MERGE (due:DueDate {value: $due_date})
            SET due.node_type = 'DueDate'
            MERGE (t)-[:DUE_ON]->(due)

            FOREACH (_s IN CASE WHEN $status IS NOT NULL THEN [1] ELSE [] END |
                MERGE (s:Status {name: $status})
                SET s.node_type = 'Status'
                MERGE (due)-[:HAS_STATUS]->(s)

                FOREACH (_p IN CASE WHEN $priority IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (p:Priority {level: $priority})
                    SET p.node_type = 'Priority'
                    MERGE (s)-[:HAS_PRIORITY]->(p)
                )
            )
        )

        // Description
        FOREACH (_ IN CASE WHEN $description IS NOT NULL THEN [1] ELSE [] END |
            MERGE (desc:Description {text: $description})
            SET desc.node_type = 'Description'
            MERGE (t)-[:HAS_DESCRIPTION]->(desc)
        )

        // EmailMessage
        MERGE (email:EmailMessage {source: $email_source})
        SET email.node_type = 'EmailMessage'
        MERGE (t)-[:LINKED_TO]->(email)

        // Sender and SpamFlag
        FOREACH (_ IN CASE WHEN $sender IS NOT NULL THEN [1] ELSE [] END |
            MERGE (sender:Sender {email: $sender})
            SET sender.node_type = 'Sender'
            MERGE (t)-[:SENT_BY]->(sender)

            FOREACH (_spam IN CASE WHEN $spam = true THEN [1] ELSE [] END |
                MERGE (spamFlag:SpamFlag {label: "Spam"})
                SET spamFlag.node_type = 'SpamFlag'
                MERGE (sender)-[:IS_SPAM_SOURCE]->(spamFlag)
            )
        )

        // Assignee and SpamFlag
        FOREACH (_ IN CASE WHEN $assignee IS NOT NULL THEN [1] ELSE [] END |
            MERGE (assignee:Assignee {email: $assignee})
            SET assignee.node_type = 'Assignee'
            MERGE (t)-[:ASSIGNED_TO]->(assignee)

            FOREACH (_spam IN CASE WHEN $spam = true THEN [1] ELSE [] END |
                MERGE (spamFlag:SpamFlag {label: "Spam"})
                SET spamFlag.node_type = 'SpamFlag'
                MERGE (assignee)-[:IS_SPAMMER]->(spamFlag)
            )
        )
    """, 
    name=task.get("name"),
    description=task.get("description"),
    email_source=task.get("email_source"),
    due_date=task.get("due_date"),
    received_date=task.get("received_date"),
    topic=task.get("topic"),
    status=task.get("status"),
    priority=task.get("priority"),
    sender=task.get("sender"),
    assignee=task.get("assignee"),
    spam=task.get("spam", False)
    )
