"""
Graph Builder Module
Creates and manages Neo4j graph database for email relationships
Supports Neo4j Aura cloud and local instances
"""

from neo4j import GraphDatabase
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import os
from dataclasses import asdict

from email_parser import ParsedEmail
from nlp_processor import EmailInsights, ExtractedEntity, ExtractedTask

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class GraphBuilder:
    """Builds and manages Neo4j graph database (supports Aura cloud)"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        Initialize Neo4j connection
        
        Credentials are loaded in this order:
        1. Parameters passed to constructor
        2. Environment variables (.env file)
        3. Default local values
        
        Environment variables:
        - NEO4J_URI: Connection URI
        - NEO4J_USERNAME: Username (usually 'neo4j')
        - NEO4J_PASSWORD: Password
        """
        
        # Load from environment if not provided
        if not uri:
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        if not username:
            username = os.getenv('NEO4J_USERNAME', 'neo4j')
        if not password:
            password = os.getenv('NEO4J_PASSWORD', 'password')
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            # Determine connection type
            connection_type = "Aura Cloud" if "neo4j.io" in uri else "Local"
            print(f"✓ Connected to Neo4j {connection_type}")
            
            self._create_constraints()
            
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            print("Connection details:")
            print(f"  URI: {uri}")
            print(f"  Username: {username}")
            print("  Password: [hidden]")
            print("\n💡 For Neo4j Aura, make sure your .env file contains:")
            print("  NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io")
            print("  NEO4J_USERNAME=neo4j")
            print("  NEO4J_PASSWORD=your-password")
            raise
    
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def _create_constraints(self):
        """Create database constraints and indexes"""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT person_email IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE",
                "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.message_id IS UNIQUE",
                "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (tp:Topic) REQUIRE tp.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint creation note: {e}")
    
    def add_email(self, email: ParsedEmail, insights: EmailInsights) -> None:
        """Add email and its insights to the graph"""
        with self.driver.session() as session:
            # Create email node
            session.run("""
                MERGE (e:Email {message_id: $message_id})
                SET e.subject = $subject,
                    e.date = datetime($date),
                    e.body = $body,
                    e.folder_path = $folder_path,
                    e.summary = $summary,
                    e.sentiment = $sentiment,
                    e.urgency_score = $urgency_score
            """, 
                message_id=email.message_id,
                subject=email.subject,
                date=email.date.isoformat() if email.date else None,
                body=email.body,
                folder_path=email.folder_path,
                summary=insights.summary,
                sentiment=insights.sentiment,
                urgency_score=insights.urgency_score
            )
            
            # Add sender
            if email.from_email:
                self._add_person(session, email.from_email)
                session.run("""
                    MATCH (p:Person {email: $from_email})
                    MATCH (e:Email {message_id: $message_id})
                    MERGE (p)-[:SENT]->(e)
                """, from_email=email.from_email, message_id=email.message_id)
            
            # Add recipients
            for to_email in email.to_emails:
                self._add_person(session, to_email)
                session.run("""
                    MATCH (p:Person {email: $to_email})
                    MATCH (e:Email {message_id: $message_id})
                    MERGE (e)-[:SENT_TO]->(p)
                """, to_email=to_email, message_id=email.message_id)
            
            # Add CC recipients
            for cc_email in email.cc_emails:
                self._add_person(session, cc_email)
                session.run("""
                    MATCH (p:Person {email: $cc_email})
                    MATCH (e:Email {message_id: $message_id})
                    MERGE (e)-[:CC]->(p)
                """, cc_email=cc_email, message_id=email.message_id)
            
            # Add entities
            for entity in insights.entities:
                self._add_entity(session, entity, email.message_id)
            
            # Add tasks
            for i, task in enumerate(insights.tasks):
                self._add_task(session, task, email.message_id, i)
            
            # Add topics
            for topic_name, weight in insights.topics:
                self._add_topic(session, topic_name, weight, email.message_id)
    
    def _add_person(self, session, email_addr: str) -> None:
        """Add person node"""
        # Extract name from email if possible
        name = email_addr.split('@')[0].replace('.', ' ').title()
        domain = email_addr.split('@')[1] if '@' in email_addr else ''
        
        session.run("""
            MERGE (p:Person {email: $email})
            SET p.name = COALESCE(p.name, $name),
                p.domain = $domain
        """, email=email_addr, name=name, domain=domain)
    
    def _add_entity(self, session, entity: ExtractedEntity, message_id: str) -> None:
        """Add entity and link to email"""
        if entity.label == 'PERSON':
            session.run("""
                MERGE (p:Person {name: $name})
                WITH p
                MATCH (e:Email {message_id: $message_id})
                MERGE (e)-[:MENTIONS_PERSON]->(p)
            """, name=entity.text, message_id=message_id)
            
        elif entity.label == 'ORG':
            session.run("""
                MERGE (o:Organization {name: $name})
                WITH o
                MATCH (e:Email {message_id: $message_id})
                MERGE (e)-[:MENTIONS_ORG]->(o)
            """, name=entity.text, message_id=message_id)
            
        elif entity.label in ['GPE', 'LOC']:
            session.run("""
                MERGE (l:Location {name: $name})
                WITH l
                MATCH (e:Email {message_id: $message_id})
                MERGE (e)-[:MENTIONS_LOCATION]->(l)
            """, name=entity.text, message_id=message_id)
    
    def _add_task(self, session, task: ExtractedTask, message_id: str, task_index: int) -> None:
        """Add task and link to email"""
        task_id = f"{message_id}_{task_index}"
        
        session.run("""
            MERGE (t:Task {id: $task_id})
            SET t.description = $description,
                t.priority = $priority,
                t.due_date = $due_date,
                t.confidence = $confidence,
                t.context = $context
            WITH t
            MATCH (e:Email {message_id: $message_id})
            MERGE (e)-[:CONTAINS_TASK]->(t)
        """, 
            task_id=task_id,
            description=task.task_description,
            priority=task.priority,
            due_date=task.due_date.isoformat() if task.due_date else None,
            confidence=task.confidence,
            context=task.context,
            message_id=message_id
        )
        
        # Link to assignee if available
        if task.assignee and task.assignee != "recipient":
            session.run("""
                MATCH (t:Task {id: $task_id})
                MERGE (p:Person {email: $assignee})
                MERGE (t)-[:ASSIGNED_TO]->(p)
            """, task_id=task_id, assignee=task.assignee)
    
    def _add_topic(self, session, topic_name: str, weight: float, message_id: str) -> None:
        """Add topic and link to email"""
        session.run("""
            MERGE (tp:Topic {name: $topic_name})
            WITH tp
            MATCH (e:Email {message_id: $message_id})
            MERGE (e)-[r:HAS_TOPIC]->(tp)
            SET r.weight = $weight
        """, topic_name=topic_name, weight=weight, message_id=message_id)
    
    def get_person_network(self, email: str, depth: int = 2) -> Dict:
        """Get network of people connected to a person"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {email: $email})
                CALL apoc.path.subgraphNodes(p, {
                    relationshipFilter: "SENT|SENT_TO|CC",
                    minLevel: 0,
                    maxLevel: $depth
                }) YIELD node
                RETURN node
            """, email=email, depth=depth)
            
            nodes = [record["node"] for record in result]
            return {"nodes": nodes}
    
    def get_email_thread(self, subject: str) -> List[Dict]:
        """Get emails in a thread by subject"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Email)
                WHERE e.subject CONTAINS $subject
                RETURN e
                ORDER BY e.date
            """, subject=subject)
            
            return [dict(record["e"]) for record in result]
    
    def get_tasks_by_person(self, email: str) -> List[Dict]:
        """Get all tasks assigned to a person"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {email: $email})<-[:ASSIGNED_TO]-(t:Task)
                OPTIONAL MATCH (t)<-[:CONTAINS_TASK]-(e:Email)
                RETURN t, e
                ORDER BY t.due_date
            """, email=email)
            
            tasks = []
            for record in result:
                task = dict(record["t"])
                email_info = dict(record["e"]) if record["e"] else None
                task["email"] = email_info
                tasks.append(task)
            
            return tasks
    
    def get_organization_interactions(self, org_name: str) -> Dict:
        """Get all interactions involving an organization"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (o:Organization {name: $org_name})<-[:MENTIONS_ORG]-(e:Email)
                MATCH (e)<-[:SENT]-(sender:Person)
                OPTIONAL MATCH (e)-[:SENT_TO]->(recipient:Person)
                RETURN e, sender, collect(recipient) as recipients
                ORDER BY e.date DESC
                LIMIT 50
            """, org_name=org_name)
            
            interactions = []
            for record in result:
                interaction = {
                    "email": dict(record["e"]),
                    "sender": dict(record["sender"]),
                    "recipients": [dict(r) for r in record["recipients"]]
                }
                interactions.append(interaction)
            
            return {"interactions": interactions}
    
    def get_topic_analysis(self, limit: int = 10) -> List[Dict]:
        """Get most discussed topics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (tp:Topic)<-[r:HAS_TOPIC]-(e:Email)
                RETURN tp.name as topic, 
                       count(e) as email_count,
                       avg(r.weight) as avg_weight
                ORDER BY email_count DESC
                LIMIT $limit
            """, limit=limit)
            
            return [dict(record) for record in result]
    
    def get_communication_patterns(self, start_date: str, end_date: str) -> Dict:
        """Analyze communication patterns in date range"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (sender:Person)-[:SENT]->(e:Email)-[:SENT_TO]->(recipient:Person)
                WHERE e.date >= datetime($start_date) AND e.date <= datetime($end_date)
                RETURN sender.email as sender, 
                       recipient.email as recipient,
                       count(e) as email_count
                ORDER BY email_count DESC
                LIMIT 50
            """, start_date=start_date, end_date=end_date)
            
            patterns = [dict(record) for record in result]
            return {"patterns": patterns}
    
    def search_emails(self, query: str, limit: int = 20) -> List[Dict]:
        """Search emails by content"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Email)
                WHERE e.subject CONTAINS $query OR e.body CONTAINS $query
                OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
                RETURN e, sender
                ORDER BY e.date DESC
                LIMIT $limit
            """, query=query, limit=limit)
            
            emails = []
            for record in result:
                email_data = dict(record["e"])
                sender_data = dict(record["sender"]) if record["sender"] else None
                email_data["sender"] = sender_data
                emails.append(email_data)
            
            return emails