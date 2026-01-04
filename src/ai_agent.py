"""
AI Agent Module
Main orchestrator that coordinates all components
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from email_parser import EmailParser, ParsedEmail
from nlp_processor import NLPProcessor, EmailInsights
from graph_builder import GraphBuilder
from task_manager import TaskManager, StructuredTask


class EmailIntelligenceAgent:
    """Main AI agent that orchestrates email processing"""
    
    def __init__(self, maildir_path: str = "maildir", 
                 neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        
        self.email_parser = EmailParser(maildir_path)
        self.nlp_processor = NLPProcessor()
        self.task_manager = TaskManager()
        
        # Initialize graph database (optional) - reads from .env if params not provided
        self.graph_builder = None
        try:
            self.graph_builder = GraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
            print("Connected to Neo4j database")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            print("Continuing without graph database...")
        
        self.processed_emails: List[ParsedEmail] = []
        self.email_insights: Dict[str, EmailInsights] = {}
        self.processing_stats = {
            'total_emails': 0,
            'processed_emails': 0,
            'extracted_tasks': 0,
            'identified_entities': 0,
            'processing_errors': 0
        }
    
    def initialize(self):
        """Initialize NLP models"""
        print("Initializing NLP models...")
        self.nlp_processor.initialize_models()
        print("Agent initialization complete!")
    
    def process_emails(self, limit: Optional[int] = None, 
                      users: Optional[List[str]] = None) -> Dict:
        """Process emails and extract insights"""
        print("Starting email processing...")
        
        # Parse emails
        if users:
            emails = []
            for user in users:
                user_emails = self.email_parser.get_user_emails(user)
                emails.extend(user_emails)
                if limit and len(emails) >= limit:
                    emails = emails[:limit]
                    break
        else:
            emails = self.email_parser.parse_all_emails(limit)
        
        self.processing_stats['total_emails'] = len(emails)
        print(f"Found {len(emails)} emails to process")
        
        # Process each email
        for i, email in enumerate(emails):
            try:
                # Extract insights
                insights = self.nlp_processor.process_email(
                    email.body, email.subject, email.from_email
                )
                
                # Process tasks
                structured_tasks = self.task_manager.process_extracted_tasks(
                    insights.tasks, email.message_id, email.from_email
                )
                
                # Store results
                self.processed_emails.append(email)
                self.email_insights[email.message_id] = insights
                
                # Add to graph database if available
                if self.graph_builder:
                    try:
                        self.graph_builder.add_email(email, insights)
                    except Exception as e:
                        print(f"Graph database error for email {email.message_id}: {e}")
                
                # Update stats
                self.processing_stats['processed_emails'] += 1
                self.processing_stats['extracted_tasks'] += len(insights.tasks)
                self.processing_stats['identified_entities'] += len(insights.entities)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(emails)} emails...")
                    
            except Exception as e:
                print(f"Error processing email {email.message_id}: {e}")
                self.processing_stats['processing_errors'] += 1
        
        # Extract topics from all emails
        self._extract_global_topics()
        
        print("Email processing complete!")
        return self.get_processing_summary()
    
    def _extract_global_topics(self):
        """Extract topics across all processed emails"""
        if not self.processed_emails:
            return
        
        print("Extracting global topics...")
        
        # Collect all email texts
        texts = []
        for email in self.processed_emails:
            full_text = f"{email.subject} {email.body}".strip()
            if len(full_text) > 50:  # Skip very short emails
                texts.append(full_text)
        
        # Extract topics
        topics = self.nlp_processor.extract_topics(texts, n_topics=10)
        
        # Update insights with topics
        for email in self.processed_emails:
            if email.message_id in self.email_insights:
                self.email_insights[email.message_id].topics = topics
        
        print(f"Extracted {len(topics)} global topics")
    
    def get_processing_summary(self) -> Dict:
        """Get summary of processing results"""
        task_report = self.task_manager.generate_task_report()
        
        return {
            'processing_stats': self.processing_stats,
            'task_analytics': task_report,
            'top_topics': self._get_top_topics(),
            'communication_summary': self._get_communication_summary()
        }
    
    def _get_top_topics(self) -> List[Tuple[str, float]]:
        """Get most common topics"""
        if not self.email_insights:
            return []
        
        # Get topics from first email (they're global)
        first_insights = next(iter(self.email_insights.values()))
        return first_insights.topics[:5]
    
    def _get_communication_summary(self) -> Dict:
        """Get communication patterns summary"""
        if not self.processed_emails:
            return {}
        
        # Count emails by sender
        sender_counts = {}
        recipient_counts = {}
        
        for email in self.processed_emails:
            sender_counts[email.from_email] = sender_counts.get(email.from_email, 0) + 1
            
            for recipient in email.to_emails:
                recipient_counts[recipient] = recipient_counts.get(recipient, 0) + 1
        
        # Get top senders and recipients
        top_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_recipients = sorted(recipient_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_unique_senders': len(sender_counts),
            'total_unique_recipients': len(recipient_counts),
            'top_senders': top_senders,
            'top_recipients': top_recipients
        }
    
    def search_emails(self, query: str, limit: int = 20) -> List[Dict]:
        """Search emails by content"""
        results = []
        query_lower = query.lower()
        
        for email in self.processed_emails:
            if (query_lower in email.subject.lower() or 
                query_lower in email.body.lower()):
                
                insights = self.email_insights.get(email.message_id)
                result = {
                    'email': email.to_dict(),
                    'insights': {
                        'summary': insights.summary if insights else '',
                        'sentiment': insights.sentiment if insights else 'neutral',
                        'urgency_score': insights.urgency_score if insights else 0.0,
                        'task_count': len(insights.tasks) if insights else 0
                    }
                }
                results.append(result)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_person_insights(self, email_address: str) -> Dict:
        """Get insights about a specific person"""
        sent_emails = [e for e in self.processed_emails if e.from_email == email_address]
        received_emails = [e for e in self.processed_emails if email_address in e.to_emails]
        
        # Get tasks assigned to this person
        assigned_tasks = self.task_manager.get_tasks_by_assignee(email_address)
        
        # Calculate communication patterns
        communication_partners = {}
        for email in sent_emails:
            for recipient in email.to_emails:
                communication_partners[recipient] = communication_partners.get(recipient, 0) + 1
        
        return {
            'email_address': email_address,
            'emails_sent': len(sent_emails),
            'emails_received': len(received_emails),
            'tasks_assigned': len(assigned_tasks),
            'top_communication_partners': sorted(communication_partners.items(), 
                                               key=lambda x: x[1], reverse=True)[:5],
            'recent_tasks': [task.to_dict() for task in assigned_tasks[-5:]]
        }
    
    def export_results(self, output_dir: str = "output"):
        """Export all results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Exporting results to {output_dir}...")
        
        # Export processed emails
        emails_data = [email.to_dict() for email in self.processed_emails]
        with open(output_path / "processed_emails.json", 'w') as f:
            json.dump(emails_data, f, indent=2, default=str)
        
        # Export insights
        insights_data = {}
        for msg_id, insights in self.email_insights.items():
            insights_data[msg_id] = {
                'entities': [{'text': e.text, 'label': e.label} for e in insights.entities],
                'topics': insights.topics,
                'tasks': [{'description': t.task_description, 'assignee': t.assignee, 
                          'due_date': t.due_date.isoformat() if t.due_date else None,
                          'priority': t.priority} for t in insights.tasks],
                'summary': insights.summary,
                'sentiment': insights.sentiment,
                'urgency_score': insights.urgency_score
            }
        
        with open(output_path / "email_insights.json", 'w') as f:
            json.dump(insights_data, f, indent=2, default=str)
        
        # Export tasks
        self.task_manager.export_to_json(str(output_path / "structured_tasks.json"))
        self.task_manager.export_to_calendar(str(output_path / "task_calendar.json"))
        self.task_manager.export_to_todo_list(str(output_path / "todo_list.json"))
        
        # Export summary report
        summary = self.get_processing_summary()
        with open(output_path / "processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create CSV for analysis
        self._export_csv_analysis(output_path)
        
        print("Export complete!")
    
    def _export_csv_analysis(self, output_path: Path):
        """Export data in CSV format for analysis"""
        # Email analysis CSV
        email_data = []
        for email in self.processed_emails:
            insights = self.email_insights.get(email.message_id)
            email_data.append({
                'message_id': email.message_id,
                'date': email.date,
                'from_email': email.from_email,
                'to_count': len(email.to_emails),
                'cc_count': len(email.cc_emails),
                'subject': email.subject,
                'body_length': len(email.body),
                'sentiment': insights.sentiment if insights else 'neutral',
                'urgency_score': insights.urgency_score if insights else 0.0,
                'task_count': len(insights.tasks) if insights else 0,
                'entity_count': len(insights.entities) if insights else 0
            })
        
        df_emails = pd.DataFrame(email_data)
        df_emails.to_csv(output_path / "email_analysis.csv", index=False)
        
        # Task analysis CSV
        task_data = []
        for task in self.task_manager.tasks.values():
            task_data.append({
                'task_id': task.id,
                'title': task.title,
                'description': task.description,
                'assignee': task.assignee,
                'creator': task.creator,
                'due_date': task.due_date,
                'priority': task.priority.value,
                'status': task.status.value,
                'confidence': task.confidence,
                'estimated_duration': task.estimated_duration,
                'tags': ','.join(task.tags)
            })
        
        if task_data:
            df_tasks = pd.DataFrame(task_data)
            df_tasks.to_csv(output_path / "task_analysis.csv", index=False)
    
    def close(self):
        """Clean up resources"""
        if self.graph_builder:
            self.graph_builder.close()