#!/usr/bin/env python3
"""
Neo4j Graph Writer - Replaces NetworkX with Neo4j for better performance
Converts extracted task JSON to Neo4j graph database
"""

import os
import logging
from typing import List, Dict, Optional
from neo4j import GraphDatabase, Driver

# Handle dotenv import gracefully for deployment environments
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class Neo4jGraphWriter:
    """Neo4j graph writer for task data."""

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver: Optional[Driver] = None

    def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        try:
            drv: Driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            # Test connection
            with drv.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Connected to Neo4j for graph writing")
            self.driver = drv
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def clear_graph(self):
        """Clear all nodes, relationships, constraints, and indexes from the graph."""
        if not self.driver:
            if not self.connect():
                return False
        try:
            driver = self.driver
            assert driver is not None
            with driver.session() as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                # Drop all constraints
                constraints = session.run("SHOW CONSTRAINTS")
                for record in constraints:
                    try:
                        session.run(f"DROP CONSTRAINT {record['name']}")
                    except Exception as e:
                        logger.debug(f"Could not drop constraint {record['name']}: {e}")
                # Drop all indexes
                indexes = session.run("SHOW INDEXES")
                for record in indexes:
                    try:
                        session.run(f"DROP INDEX {record['name']}")
                    except Exception as e:
                        logger.debug(f"Could not drop index {record['name']}: {e}")
            logger.info("✅ Neo4j graph and schema fully cleared")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing Neo4j graph: {e}")
            return False

    def create_constraints(self):
        """Create uniqueness constraints for better performance."""
        if not self.driver:
            if not self.connect():
                return False

        constraints = [
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT task_name IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT role_name IF NOT EXISTS FOR (r:Role) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT dept_name IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT date_name IF NOT EXISTS FOR (d:Date) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT summary_name IF NOT EXISTS FOR (s:Summary) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT email_name IF NOT EXISTS FOR (e:EmailIndex) REQUIRE e.name IS UNIQUE",
        ]

        try:
            driver = self.driver
            assert driver is not None
            with driver.session() as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        logger.debug(f"Constraint already exists or error: {e}")
            logger.info("✅ Neo4j constraints created")
            return True
        except Exception as e:
            logger.error(f"❌ Error creating constraints: {e}")
            return False

    # Old write_tasks_to_graph method removed - using write_tasks_from_table for flat structure
    # Old _clean_dict method removed - using flat structure

    # Old _write_single_topic and _create_person_hierarchy methods removed - using flat structure

    def get_graph_stats(self) -> Dict:
        """Get statistics about the current graph."""
        if not self.driver:
            if not self.connect():
                return {}

        try:
            driver = self.driver
            assert driver is not None
            with driver.session() as session:
                # Count nodes by label
                stats = {}
                labels = [
                    "Topic",
                    "Task",
                    "Person",
                    "Role",
                    "Department",
                    "Organization",
                    "Date",
                    "Summary",
                    "EmailIndex",
                ]

                for label in labels:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    stats[label] = result.single()["count"]

                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats["Relationships"] = rel_result.single()["count"]

                return stats
        except Exception as e:
            logger.error(f"❌ Error getting graph stats: {e}")
            return {}

    def write_tasks_from_table(
        self, tasks_data: List[Dict], clear_existing: bool = False
    ) -> bool:
        """Write tasks from flat table structure to Neo4j."""
        if not self.connect():
            return False

        try:
            if clear_existing:
                self.clear_graph()
            self.create_constraints()

            with self.driver.session() as session:
                for task_row in tasks_data:
                    self._write_single_task_from_table(session, task_row)

            logger.info(
                f"✅ Successfully wrote {len(tasks_data)} tasks to Neo4j from table structure"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error writing tasks to Neo4j: {e}")
            return False
        finally:
            self.close()

    def _write_single_task_from_table(self, session, task_row: Dict):
        """Write a single task from flat table structure to Neo4j."""

        # Extract task data from flat structure
        task_name = task_row.get("task_name", "Unknown Task")
        task_description = task_row.get("task_description", "")
        topic_name = task_row.get("topic", "Unknown Topic")
        sender = task_row.get("sender", "")
        assigned_to = task_row.get("assigned_to", "")
        due_date = task_row.get("due_date")
        received_date = task_row.get("received_date")
        status = task_row.get("status", "not started")
        priority_level = task_row.get("priority_level", "")
        message_id = task_row.get("message_id", "")

        # Create topic node
        session.run(
            """
            MERGE (t:Topic {name: $name})
        """,
            name=topic_name,
        )

        # Create task node with flat structure
        session.run(
            """
            MERGE (task:Task {name: $name})
            SET task.description = $description,
                task.status = $status,
                task.priority_level = $priority_level,
                task.message_id = $message_id,
                task.spam = $spam
        """,
            name=task_name,
            description=task_description,
            status=status,
            priority_level=priority_level,
            message_id=message_id,
            spam=task_row.get("spam", False),
        )

        # Link topic to task
        session.run(
            """
            MATCH (t:Topic {name: $topic_name})
            MATCH (task:Task {name: $task_name})
            MERGE (t)-[:HAS_TASK]->(task)
        """,
            topic_name=topic_name,
            task_name=task_name,
        )

        # Create sender person
        if sender:
            session.run(
                """
                MERGE (p:Person {name: $name})
                SET p.role = 'Sender'
            """,
                name=sender,
            )

            # Link sender to task
            session.run(
                """
                MATCH (task:Task {name: $task_name})
                MATCH (p:Person {name: $sender})
                MERGE (p)-[:SENT_TASK]->(task)
            """,
                task_name=task_name,
                sender=sender,
            )

        # Create assigned person
        if assigned_to:
            session.run(
                """
                MERGE (p:Person {name: $name})
                SET p.role = 'Assigned'
            """,
                name=assigned_to,
            )

            # Link assigned person to task
            session.run(
                """
                MATCH (task:Task {name: $task_name})
                MATCH (p:Person {name: $assigned_to})
                MERGE (task)-[:ASSIGNED_TO]->(p)
            """,
                task_name=task_name,
                assigned_to=assigned_to,
            )

        # Create date nodes
        if due_date:
            session.run(
                """
                MERGE (d:Date {name: $date})
                WITH d
                MATCH (task:Task {name: $task_name})
                MERGE (task)-[:DUE_ON]->(d)
            """,
                date=str(due_date),
                task_name=task_name,
            )

        if received_date:
            session.run(
                """
                MERGE (d:Date {name: $date})
                WITH d
                MATCH (task:Task {name: $task_name})
                MERGE (task)-[:RECEIVED_ON]->(d)
            """,
                date=str(received_date),
                task_name=task_name,
            )

        # Create email index node if message_id exists
        if message_id:
            session.run(
                """
                MERGE (e:EmailIndex {name: $message_id})
                WITH e
                MATCH (task:Task {name: $task_name})
                MERGE (task)-[:LINKED_TO]->(e)
                        """,
                message_id=message_id,
                task_name=task_name,
            )