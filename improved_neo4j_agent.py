#!/usr/bin/env python3
"""
Improved Neo4j Email Agent
Better regex patterns and filtering for higher quality extraction
No ML dependencies to avoid mutex issues
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Force single-threaded execution
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    print("⚠️  Neo4j driver not installed. Install with: pip install neo4j")
    NEO4J_AVAILABLE = False


class ImprovedNeo4jAgent:
    """Improved email intelligence agent with better extraction patterns"""
    
    def __init__(self):
        self.emails = []
        self.tasks = []
        self.entities = []
        self.stats = {
            'start_time': datetime.now(),
            'emails_processed': 0,
            'tasks_extracted': 0,
            'entities_found': 0,
            'neo4j_nodes_created': 0,
            'neo4j_relationships_created': 0,
            'errors': 0
        }
        
        # Initialize Neo4j connection
        self.driver = None
        self._connect_neo4j()
        
        # Load business vocabulary for better extraction
        self._load_business_terms()
    
    def _load_business_terms(self):
        """Load business-specific terms for better task detection"""
        self.action_verbs = {
            'schedule', 'arrange', 'organize', 'coordinate', 'plan', 'book',
            'review', 'check', 'verify', 'confirm', 'validate', 'approve', 'audit',
            'send', 'provide', 'deliver', 'submit', 'prepare', 'create', 'draft', 'write',
            'update', 'revise', 'modify', 'change', 'fix', 'correct', 'adjust',
            'meet', 'call', 'discuss', 'talk', 'contact', 'reach', 'follow',
            'analyze', 'evaluate', 'assess', 'investigate', 'research', 'study',
            'implement', 'execute', 'complete', 'finish', 'deliver', 'launch'
        }
        
        self.business_objects = {
            'report', 'document', 'presentation', 'proposal', 'contract', 'agreement',
            'meeting', 'conference', 'call', 'session', 'workshop', 'training',
            'project', 'task', 'deliverable', 'milestone', 'deadline', 'timeline',
            'budget', 'forecast', 'analysis', 'review', 'audit', 'assessment',
            'strategy', 'plan', 'roadmap', 'schedule', 'calendar', 'agenda'
        }
        
        self.urgency_indicators = {
            'urgent', 'asap', 'critical', 'immediately', 'emergency', 'rush',
            'deadline', 'due', 'overdue', 'priority', 'important', 'crucial'
        }
    
    def _connect_neo4j(self):
        """Connect to Neo4j Aura"""
        if not NEO4J_AVAILABLE:
            print("❌ Neo4j driver not available")
            return
        
        try:
            uri = os.getenv('NEO4J_URI')
            username = os.getenv('NEO4J_USERNAME')
            password = os.getenv('NEO4J_PASSWORD')
            
            if not all([uri, username, password]):
                print("❌ Neo4j credentials not found in .env file")
                return
            
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful' as message")
                message = result.single()["message"]
                print(f"✅ Neo4j Aura connected: {message}")
                
                # Clear existing data
                print("🧹 Clearing existing data...")
                session.run("MATCH (n) DETACH DELETE n")
                
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self.driver = None
    
    def load_emails(self, maildir_path: str = "maildir", limit: int = 30):
        """Load emails from Enron dataset"""
        print(f"📧 Loading emails from {maildir_path}...")
        
        maildir = Path(maildir_path)
        if not maildir.exists():
            print(f"❌ Directory {maildir_path} not found!")
            return False
        
        # Target users with good business communication
        target_users = ["kaminski-v", "beck-s", "allen-p", "lay-k", "skilling-j"]
        
        count = 0
        for user_dir in maildir.iterdir():
            if not user_dir.is_dir() or user_dir.name not in target_users:
                continue
            
            print(f"   📁 Processing {user_dir.name}...")
            
            # Look for emails in various folders
            for subfolder in ["sent_items", "inbox", "_sent_mail", "sent"]:
                folder_path = user_dir / subfolder
                if folder_path.exists():
                    for email_file in folder_path.iterdir():
                        if email_file.is_file():
                            email = self._parse_email(email_file)
                            if email and self._is_business_email(email):
                                self.emails.append(email)
                                count += 1
                                
                                if count >= limit:
                                    break
                    
                    if count >= limit:
                        break
            
            if count >= limit:
                break
        
        print(f"✅ Loaded {len(self.emails)} business emails")
        return len(self.emails) > 0
    
    def _parse_email(self, file_path: Path) -> Optional[Dict]:
        """Parse single email file with better cleaning"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            email = {
                'id': f"email_{file_path.stem}",
                'file_path': str(file_path),
                'subject': '',
                'from': '',
                'to': [],
                'cc': [],
                'date': '',
                'body': '',
                'raw_content': content
            }
            
            lines = content.split('\n')
            body_start = 0
            
            # Parse headers with better extraction
            for i, line in enumerate(lines):
                line = line.strip()
                
                if line.startswith('Subject: '):
                    email['subject'] = self._clean_subject(line[9:])
                elif line.startswith('From: '):
                    email['from'] = self._extract_clean_email(line[6:])
                elif line.startswith('To: '):
                    email['to'] = self._parse_clean_recipients(line[4:])
                elif line.startswith('Cc: '):
                    email['cc'] = self._parse_clean_recipients(line[4:])
                elif line.startswith('Date: '):
                    email['date'] = line[6:].strip()
                elif line == '':
                    body_start = i + 1
                    break
            
            # Extract and clean body
            raw_body = '\n'.join(lines[body_start:]).strip()
            email['body'] = self._clean_body(raw_body)
            
            # Only keep emails with meaningful business content
            if (len(email['body']) > 50 and 
                email['subject'] and 
                email['from'] and 
                self._has_business_content(email['body'])):
                return email
            
            return None
            
        except Exception as e:
            return None
    
    def _clean_subject(self, subject: str) -> str:
        """Clean email subject"""
        # Remove common prefixes
        subject = re.sub(r'^(RE:|FW:|FWD:)\s*', '', subject, flags=re.IGNORECASE)
        return subject.strip()
    
    def _extract_clean_email(self, email_str: str) -> str:
        """Extract clean email address"""
        # Look for email in angle brackets first
        match = re.search(r'<([^>]+@[^>]+)>', email_str)
        if match:
            email = match.group(1).lower().strip()
        else:
            # Look for standalone email
            match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', email_str)
            if match:
                email = match.group(1).lower().strip()
            else:
                return email_str.strip().lower()
        
        # Filter out malformed emails
        if any(x in email for x in ['imceanotes', '+3e+40', 'enron@enron']):
            return ''
        
        return email
    
    def _parse_clean_recipients(self, recipients_str: str) -> List[str]:
        """Parse and clean recipient list"""
        recipients = []
        for recipient in recipients_str.split(','):
            email_addr = self._extract_clean_email(recipient.strip())
            if email_addr and '@' in email_addr and len(email_addr) > 5:
                recipients.append(email_addr)
        return recipients
    
    def _clean_body(self, body: str) -> str:
        """Clean email body text"""
        # Remove forwarded message headers
        body = re.sub(r'-+\s*Forwarded by.*?-+', '', body, flags=re.DOTALL)
        body = re.sub(r'-+\s*Original Message\s*-+.*?(?=\n\n|\Z)', '', body, flags=re.DOTALL)
        
        # Remove email signatures
        body = re.sub(r'\n\n.*?(?:phone|fax|cell).*?\n.*?\n', '\n\n', body, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove excessive whitespace
        body = re.sub(r'\n\s*\n\s*\n', '\n\n', body)
        body = re.sub(r'[ \t]+', ' ', body)
        
        return body.strip()
    
    def _is_business_email(self, email: Dict) -> bool:
        """Check if email contains business content"""
        text = f"{email['subject']} {email['body']}".lower()
        
        # Must contain business-related terms
        business_terms = ['meeting', 'project', 'report', 'schedule', 'deadline', 
                         'budget', 'contract', 'proposal', 'presentation', 'review',
                         'analysis', 'strategy', 'plan', 'deliverable', 'task']
        
        return any(term in text for term in business_terms)
    
    def _has_business_content(self, body: str) -> bool:
        """Check if body has substantial business content"""
        # Must have reasonable length and business terms
        if len(body) < 50:
            return False
        
        # Count business-related words
        business_word_count = 0
        words = body.lower().split()
        
        for word in words:
            if word in self.action_verbs or word in self.business_objects:
                business_word_count += 1
        
        # At least 2% of words should be business-related
        return business_word_count / len(words) >= 0.02
    
    def extract_improved_tasks(self, email: Dict) -> List[Dict]:
        """Extract tasks using improved patterns and business context"""
        full_text = f"{email['subject']} {email['body']}"
        tasks = []
        
        # Improved task patterns with business context
        task_patterns = [
            # Direct requests with action verbs
            (r'(?:please|could you|can you|would you|need you to)\s+(' + 
             '|'.join(self.action_verbs) + r')\s+([^.!?]{20,150})', 'request'),
            
            # Action items and todos
            (r'(?:action item|todo|to do|task|deliverable):\s*([^.!?]{15,150})', 'action_item'),
            
            # Business actions with objects
            (r'(?:' + '|'.join(self.action_verbs) + r')\s+(?:the\s+)?(?:' + 
             '|'.join(self.business_objects) + r')\s+([^.!?]{10,120})', 'business_action'),
            
            # Deadline-based tasks
            (r'(?:by|before|due)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}/\d{1,2}|\w+\s+\d{1,2})\s*[,:]?\s*([^.!?]{15,120})', 'deadline'),
            
            # Meeting and call scheduling
            (r'(?:schedule|arrange|set up|organize)\s+(?:a\s+)?(?:meeting|call|conference|session)\s+([^.!?]{10,100})', 'scheduling'),
            
            # Document and report requests
            (r'(?:prepare|create|draft|write|send|provide)\s+(?:a\s+)?(?:report|document|presentation|proposal|analysis)\s+([^.!?]{10,100})', 'document'),
        ]
        
        task_id = 0
        for pattern, task_type in task_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                # Handle tuple matches (from complex patterns)
                if isinstance(match, tuple):
                    task_desc = ' '.join(match).strip()
                else:
                    task_desc = match.strip()
                
                # Quality filters
                if self._is_quality_task(task_desc):
                    priority = self._determine_smart_priority(task_desc, email['subject'])
                    assignee = self._smart_assignee_detection(task_desc, email)
                    due_date = self._extract_due_date(task_desc, email['body'])
                    
                    task = {
                        'id': f"task_{email['id']}_{task_id}",
                        'description': task_desc,
                        'type': task_type,
                        'priority': priority,
                        'assignee': assignee,
                        'due_date': due_date,
                        'source_email': email['id'],
                        'from_email': email['from'],
                        'subject': email['subject'],
                        'confidence': self._calculate_task_confidence(task_desc, task_type),
                        'extracted_date': datetime.now().isoformat()
                    }
                    
                    tasks.append(task)
                    task_id += 1
        
        return tasks
    
    def _is_quality_task(self, task_desc: str) -> bool:
        """Check if extracted task is high quality"""
        # Length check
        if len(task_desc) < 15 or len(task_desc) > 200:
            return False
        
        # Must contain action verb or business object
        words = task_desc.lower().split()
        has_action = any(word in self.action_verbs for word in words)
        has_object = any(word in self.business_objects for word in words)
        
        if not (has_action or has_object):
            return False
        
        # Filter out noise
        noise_patterns = [
            r'^https?://',  # URLs
            r'^\w+@\w+',    # Email addresses
            r'^\d+$',       # Just numbers
            r'^[^a-zA-Z]*$' # No letters
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, task_desc):
                return False
        
        return True
    
    def _determine_smart_priority(self, task_desc: str, subject: str) -> str:
        """Determine priority using business context"""
        text = f"{task_desc} {subject}".lower()
        
        # High priority indicators
        high_score = 0
        for indicator in self.urgency_indicators:
            if indicator in text:
                high_score += 1
        
        # Check for deadline indicators
        if re.search(r'(?:by|before|due)\s+(?:today|tomorrow|this week|monday|tuesday|wednesday|thursday|friday)', text):
            high_score += 2
        
        # Check for executive involvement
        if any(exec in text for exec in ['ceo', 'president', 'director', 'vp', 'vice president']):
            high_score += 1
        
        # Check for financial terms
        if re.search(r'\$[\d,]+|budget|cost|revenue|profit', text):
            high_score += 1
        
        if high_score >= 2:
            return 'high'
        elif high_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _smart_assignee_detection(self, task_desc: str, email: Dict) -> str:
        """Smart assignee detection"""
        # Look for explicit assignment
        assignment_patterns = [
            r'(?:for|assign to|assigned to|ask)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s+(?:should|will|can|needs to)'
        ]
        
        for pattern in assignment_patterns:
            match = re.search(pattern, task_desc, re.IGNORECASE)
            if match:
                return self._extract_clean_email(match.group(1))
        
        # Default logic: if email has recipients, assign to primary recipient
        if email['to']:
            return email['to'][0]
        
        # Otherwise assign to sender (self-assigned task)
        return email['from']
    
    def _extract_due_date(self, task_desc: str, email_body: str) -> Optional[str]:
        """Extract due date from task description or email"""
        text = f"{task_desc} {email_body}".lower()
        
        # Look for explicit dates
        date_patterns = [
            r'(?:by|before|due)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(?:by|before|due)\s+(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(?:by|before|due)\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)',
            r'(?:by|before|due)\s+(today|tomorrow|this week|next week)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                # Convert relative dates
                if date_str == 'today':
                    return datetime.now().isoformat()
                elif date_str == 'tomorrow':
                    return (datetime.now() + timedelta(days=1)).isoformat()
                elif date_str == 'this week':
                    return (datetime.now() + timedelta(days=7)).isoformat()
                else:
                    return date_str
        
        return None
    
    def _calculate_task_confidence(self, task_desc: str, task_type: str) -> float:
        """Calculate confidence score for task extraction"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on task type
        type_boost = {
            'request': 0.3,
            'action_item': 0.4,
            'business_action': 0.2,
            'deadline': 0.3,
            'scheduling': 0.3,
            'document': 0.2
        }
        confidence += type_boost.get(task_type, 0)
        
        # Boost for business terms
        words = task_desc.lower().split()
        business_word_ratio = sum(1 for word in words if word in self.action_verbs or word in self.business_objects) / len(words)
        confidence += business_word_ratio * 0.2
        
        return min(confidence, 1.0)
    
    def extract_improved_entities(self, email: Dict) -> List[Dict]:
        """Extract entities with better patterns and filtering"""
        text = f"{email['subject']} {email['body']}"
        entities = []
        
        # Clean email addresses (filter out artifacts)
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email_addr in set(emails):
            if self._is_clean_email(email_addr):
                entities.append({
                    'text': email_addr.lower(),
                    'type': 'EMAIL',
                    'confidence': 0.9
                })
        
        # Money amounts with context
        money_patterns = [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|k|m|b))?',
            r'(?:budget|cost|price|revenue|profit):\s*\$?[\d,]+(?:\.\d{2})?'
        ]
        
        for pattern in money_patterns:
            amounts = re.findall(pattern, text, re.IGNORECASE)
            for amount in set(amounts):
                entities.append({
                    'text': amount,
                    'type': 'MONEY',
                    'confidence': 0.8
                })
        
        # Business dates (not email headers)
        business_date_patterns = [
            r'(?:meeting|deadline|due|scheduled)\s+(?:on|for)?\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:by|before)\s+(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(?:quarter|q)\s*(\d)\s*(?:20\d{2}|\d{2})'
        ]
        
        for pattern in business_date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            for date in set(dates):
                entities.append({
                    'text': date,
                    'type': 'DATE',
                    'confidence': 0.7
                })
        
        # Company names (Enron-specific)
        company_pattern = r'\b(?:Enron|EES|EPMI|ENA|Dynegy|Reliant|Calpine|Williams|El Paso)\b'
        companies = re.findall(company_pattern, text, re.IGNORECASE)
        for company in set(companies):
            entities.append({
                'text': company,
                'type': 'ORGANIZATION',
                'confidence': 0.8
            })
        
        return entities
    
    def _is_clean_email(self, email: str) -> bool:
        """Check if email address is clean and valid"""
        # Filter out common artifacts
        artifacts = ['imceanotes', '+3e+40', 'enron@enron', '+22', '+2e', '+40']
        if any(artifact in email.lower() for artifact in artifacts):
            return False
        
        # Must have proper format
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return False
        
        return True
    
    def process_all_emails(self):
        """Process all emails with improved extraction"""
        print(f"\n🧠 Processing {len(self.emails)} emails with improved patterns...")
        
        for i, email in enumerate(self.emails):
            try:
                # Extract with improved methods
                email_tasks = self.extract_improved_tasks(email)
                email_entities = self.extract_improved_entities(email)
                
                self.tasks.extend(email_tasks)
                self.entities.extend(email_entities)
                
                # Add to Neo4j
                if self.driver:
                    self._add_to_neo4j(email, email_tasks, email_entities)
                
                # Update stats
                self.stats['emails_processed'] += 1
                self.stats['tasks_extracted'] += len(email_tasks)
                self.stats['entities_found'] += len(email_entities)
                
                if (i + 1) % 5 == 0:
                    print(f"   ⚡ Processed {i + 1}/{len(self.emails)} emails...")
                
            except Exception as e:
                print(f"⚠️  Error processing email {i}: {e}")
                self.stats['errors'] += 1
        
        print(f"✅ Improved processing complete!")
    
    def _add_to_neo4j(self, email: Dict, tasks: List[Dict], entities: List[Dict]):
        """Add improved data to Neo4j"""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Create email node with metadata
                session.run("""
                    MERGE (e:Email {id: $email_id})
                    SET e.subject = $subject,
                        e.date = $date,
                        e.body_length = $body_length,
                        e.processed_date = datetime()
                """, 
                email_id=email['id'],
                subject=email['subject'],
                date=email['date'],
                body_length=len(email['body'])
                )
                self.stats['neo4j_nodes_created'] += 1
                
                # Create person nodes (clean emails only)
                if email['from']:
                    session.run("""
                        MERGE (p:Person {email: $email})
                        SET p.name = split($email, '@')[0],
                            p.domain = split($email, '@')[1]
                    """, email=email['from'])
                    self.stats['neo4j_nodes_created'] += 1
                    
                    # SENT relationship
                    session.run("""
                        MATCH (p:Person {email: $from_email})
                        WITH p
                        MATCH (e:Email {id: $email_id})
                        MERGE (p)-[:SENT]->(e)
                    """, 
                    from_email=email['from'],
                    email_id=email['id']
                    )
                    self.stats['neo4j_relationships_created'] += 1
                
                # Recipients
                for recipient in email['to']:
                    if recipient:
                        session.run("""
                            MERGE (p:Person {email: $email})
                            SET p.name = split($email, '@')[0],
                                p.domain = split($email, '@')[1]
                        """, email=recipient)
                        self.stats['neo4j_nodes_created'] += 1
                        
                        session.run("""
                            MATCH (p:Person {email: $to_email})
                            WITH p
                            MATCH (e:Email {id: $email_id})
                            MERGE (p)-[:RECEIVED]->(e)
                        """, 
                        to_email=recipient,
                        email_id=email['id']
                        )
                        self.stats['neo4j_relationships_created'] += 1
                
                # High-quality tasks
                for task in tasks:
                    session.run("""
                        CREATE (t:Task {
                            id: $task_id,
                            description: $description,
                            type: $type,
                            priority: $priority,
                            assignee: $assignee,
                            due_date: $due_date,
                            confidence: $confidence,
                            created_date: datetime()
                        })
                    """,
                    task_id=task['id'],
                    description=task['description'],
                    type=task['type'],
                    priority=task['priority'],
                    assignee=task['assignee'],
                    due_date=task['due_date'],
                    confidence=task['confidence']
                    )
                    self.stats['neo4j_nodes_created'] += 1
                    
                    # Link task to email
                    session.run("""
                        MATCH (e:Email {id: $email_id})
                        WITH e
                        MATCH (t:Task {id: $task_id})
                        MERGE (e)-[:CONTAINS_TASK]->(t)
                    """,
                    email_id=email['id'],
                    task_id=task['id']
                    )
                    self.stats['neo4j_relationships_created'] += 1
                    
                    # Link task to assignee
                    if task['assignee']:
                        session.run("""
                            MERGE (p:Person {email: $assignee})
                            WITH p
                            MATCH (t:Task {id: $task_id})
                            MERGE (p)-[:ASSIGNED_TO]->(t)
                        """,
                        assignee=task['assignee'],
                        task_id=task['id']
                        )
                        self.stats['neo4j_relationships_created'] += 1
                
                # Clean entities
                for entity in entities:
                    session.run("""
                        CREATE (ent:Entity {
                            text: $text,
                            type: $type,
                            confidence: $confidence
                        })
                    """,
                    text=entity['text'],
                    type=entity['type'],
                    confidence=entity['confidence']
                    )
                    self.stats['neo4j_nodes_created'] += 1
                    
                    # Link entity to email
                    session.run("""
                        MATCH (e:Email {id: $email_id})
                        WITH e
                        MATCH (ent:Entity {text: $text, type: $type})
                        MERGE (e)-[:MENTIONS]->(ent)
                    """,
                    email_id=email['id'],
                    text=entity['text'],
                    type=entity['type']
                    )
                    self.stats['neo4j_relationships_created'] += 1
                
        except Exception as e:
            print(f"⚠️  Neo4j error for email {email['id']}: {e}")
    
    def analyze_results(self):
        """Analyze improved results"""
        end_time = datetime.now()
        duration = (end_time - self.stats['start_time']).total_seconds()
        
        print(f"\n📊 IMPROVED ANALYSIS RESULTS")
        print("=" * 60)
        
        # Processing performance
        print(f"⏱️  Processing time: {duration:.1f} seconds")
        print(f"📧 Emails processed: {self.stats['emails_processed']}")
        print(f"✅ Tasks extracted: {self.stats['tasks_extracted']}")
        print(f"🏷️  Entities found: {self.stats['entities_found']}")
        print(f"⚡ Processing rate: {self.stats['emails_processed']/max(duration,1)*60:.0f} emails/min")
        
        # Quality metrics
        if self.stats['emails_processed'] > 0:
            print(f"\n📈 QUALITY METRICS:")
            print(f"   Tasks per email: {self.stats['tasks_extracted'] / self.stats['emails_processed']:.1f}")
            print(f"   Entities per email: {self.stats['entities_found'] / self.stats['emails_processed']:.1f}")
        
        # Task analysis
        if self.tasks:
            priority_counts = {'high': 0, 'medium': 0, 'low': 0}
            type_counts = {}
            confidence_sum = 0
            
            for task in self.tasks:
                priority_counts[task['priority']] += 1
                task_type = task.get('type', 'unknown')
                type_counts[task_type] = type_counts.get(task_type, 0) + 1
                confidence_sum += task.get('confidence', 0.5)
            
            avg_confidence = confidence_sum / len(self.tasks)
            
            print(f"\n📋 IMPROVED TASK BREAKDOWN:")
            print(f"   🔴 High priority: {priority_counts['high']}")
            print(f"   🟡 Medium priority: {priority_counts['medium']}")
            print(f"   🟢 Low priority: {priority_counts['low']}")
            print(f"   🎯 Average confidence: {avg_confidence:.2f}")
            
            print(f"\n📊 Task types:")
            for task_type, count in sorted(type_counts.items()):
                print(f"   {task_type}: {count}")
            
            # Sample high-quality tasks
            print(f"\n🎯 SAMPLE HIGH-QUALITY TASKS:")
            high_conf_tasks = sorted([t for t in self.tasks], 
                                   key=lambda x: x.get('confidence', 0), reverse=True)[:3]
            for i, task in enumerate(high_conf_tasks, 1):
                print(f"{i}. {task['description'][:80]}...")
                print(f"   Type: {task['type']} | Priority: {task['priority']} | Confidence: {task.get('confidence', 0):.2f}")
        
        # Neo4j stats
        if self.driver:
            print(f"\n🗄️  NEO4J AURA STATS")
            print(f"🔵 Nodes created: {self.stats['neo4j_nodes_created']}")
            print(f"🔗 Relationships created: {self.stats['neo4j_relationships_created']}")
            
            try:
                with self.driver.session() as session:
                    # Node counts
                    result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
                    print(f"\n📈 Node counts in Neo4j:")
                    for record in result:
                        print(f"   {record['type']}: {record['count']}")
                    
                    # Relationship counts
                    result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
                    print(f"\n🔗 Relationship counts:")
                    for record in result:
                        print(f"   {record['rel_type']}: {record['count']}")
                        
            except Exception as e:
                print(f"⚠️  Error querying Neo4j stats: {e}")
        
        if self.stats['errors'] > 0:
            print(f"⚠️  Processing errors: {self.stats['errors']}")
    
    def export_results(self, output_dir: str = "improved_results"):
        """Export improved results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📁 Exporting improved results to {output_dir}/...")
        
        # Export tasks with quality metrics
        with open(output_path / "improved_tasks.json", 'w') as f:
            json.dump(self.tasks, f, indent=2)
        
        # Export entities
        with open(output_path / "improved_entities.json", 'w') as f:
            json.dump(self.entities, f, indent=2)
        
        # Export summary with quality metrics
        summary = {
            'processing_stats': {
                'start_time': self.stats['start_time'].isoformat(),
                'emails_processed': self.stats['emails_processed'],
                'tasks_extracted': self.stats['tasks_extracted'],
                'entities_found': self.stats['entities_found'],
                'neo4j_nodes_created': self.stats['neo4j_nodes_created'],
                'neo4j_relationships_created': self.stats['neo4j_relationships_created'],
                'processing_errors': self.stats['errors']
            },
            'quality_metrics': {
                'tasks_per_email': self.stats['tasks_extracted'] / max(self.stats['emails_processed'], 1),
                'entities_per_email': self.stats['entities_found'] / max(self.stats['emails_processed'], 1),
                'avg_task_confidence': sum(t.get('confidence', 0.5) for t in self.tasks) / max(len(self.tasks), 1)
            },
            'export_date': datetime.now().isoformat()
        }
        
        with open(output_path / "improved_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Improved export complete!")
        return len(self.emails), len(self.tasks), len(self.entities)
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed")


def main():
    """Main execution with improved extraction"""
    print("🧠 IMPROVED NEO4J EMAIL AGENT")
    print("=" * 60)
    print("🎯 Better regex patterns + business context + Neo4j Aura")
    print()
    
    agent = None
    try:
        # Initialize agent
        agent = ImprovedNeo4jAgent()
        
        # Load emails
        if not agent.load_emails(limit=25):
            print("❌ Failed to load emails. Check your dataset.")
            return 1
        
        # Process with improved patterns
        agent.process_all_emails()
        
        # Analyze results
        agent.analyze_results()
        
        # Export results
        emails_count, tasks_count, entities_count = agent.export_results()
        
        # Final summary
        print(f"\n🎉 IMPROVED AGENT EXECUTION SUCCESSFUL!")
        print("=" * 60)
        print(f"📊 Processed {emails_count} business emails")
        print(f"✅ Extracted {tasks_count} improved tasks")
        print(f"🏷️  Found {entities_count} clean entities")
        print(f"🗄️  Populated Neo4j Aura with quality data")
        print(f"📂 Results saved to improved_results/")
        print()
        print("🚀 Check your Neo4j Aura browser for the improved graph!")
        print("🧠 This version uses smarter patterns and business context!")
        
    except Exception as e:
        print(f"❌ Improved agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if agent:
            agent.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())