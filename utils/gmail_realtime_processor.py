"""
Real-time Gmail Email Processor
Automatically processes new emails and extracts tasks
"""

import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from gmail_parser.gmail_service import GmailService
from gmail_parser.gmail_task_integration import filter_emails_for_task_extraction
from utils.langgraph_dag import run_extraction_pipeline
from utils.embedding import embed_dataframe
import streamlit as st

class GmailRealtimeProcessor:
    """Real-time email processor for automatic task extraction"""
    
    def __init__(self, gmail_service: GmailService):
        self.gmail_service = gmail_service
        self.last_processed_id = None
        self.is_running = False
        self.processor_thread = None
        self.callback = None
        self.processing_interval = 300  # 5 minutes default
        
    def start_monitoring(self, callback: Optional[Callable] = None, interval: int = 300):
        """
        Start monitoring for new emails
        
        Args:
            callback: Function to call when new tasks are extracted
            interval: Check interval in seconds (default 5 minutes)
        """
        if self.is_running:
            return False
            
        self.callback = callback
        self.processing_interval = interval
        self.is_running = True
        
        # Start monitoring in a separate thread
        self.processor_thread = threading.Thread(target=self._monitor_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        return True
    
    def stop_monitoring(self):
        """Stop monitoring for new emails"""
        self.is_running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Get recent unread emails
                new_emails = self.gmail_service.get_unread_emails(max_results=50)
                
                if new_emails:
                    # Filter emails for task extraction
                    filtered_emails = filter_emails_for_task_extraction(new_emails)
                    
                    if filtered_emails:
                        # Process emails for task extraction
                        extracted_tasks = self._process_emails_for_tasks(filtered_emails)
                        
                        # Call callback if provided
                        if self.callback and extracted_tasks:
                            self.callback(extracted_tasks)
                
                # Wait before next check
                time.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _process_emails_for_tasks(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process emails through the task extraction pipeline
        
        Args:
            emails: List of Gmail email dictionaries
            
        Returns:
            List of extracted task results
        """
        try:
            # Convert Gmail emails to DataFrame format
            from gmail_parser.gmail_task_integration import gmail_to_dataframe
            df = gmail_to_dataframe(emails)
            
            if df.empty:
                return []
            
            # Create embeddings for similarity search
            index, all_chunks = embed_dataframe(df)
            
            # Process each email through extraction pipeline
            results = []
            
            for index, row in df.iterrows():
                try:
                    # Convert row to dict for processing
                    email_row = row.to_dict()
                    
                    # Add email index for tracking
                    email_row['email_index'] = str(index)
                    
                    # Run extraction pipeline
                    extraction_result = run_extraction_pipeline(
                        email_row=email_row,
                        faiss_index=index,
                        all_chunks=all_chunks,
                        email_index=str(index)
                    )
                    
                    # Format result
                    result = {
                        'email_index': str(index),
                        'subject': email_row.get('Subject', ''),
                        'sender': email_row.get('From', ''),
                        'date': email_row.get('Date', ''),
                        'extraction_result': extraction_result,
                        'status': 'processed',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    # Handle processing errors
                    result = {
                        'email_index': str(index),
                        'subject': email_row.get('Subject', 'Unknown'),
                        'sender': email_row.get('From', 'Unknown'),
                        'date': email_row.get('Date', ''),
                        'status': 'error',
                        'message': f'Processing failed: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error processing emails for tasks: {e}")
            return []
    
    def process_specific_emails(self, email_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Process specific emails by their IDs
        
        Args:
            email_ids: List of Gmail message IDs
            
        Returns:
            List of extracted task results
        """
        try:
            # Get specific emails
            emails = []
            for email_id in email_ids:
                try:
                    msg = self.gmail_service.service.users().messages().get(
                        userId='me', 
                        id=email_id,
                        format='full'
                    ).execute()
                    
                    # Extract email data
                    headers = msg['payload']['headers']
                    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                    date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                    
                    # Extract body
                    body = self.gmail_service._get_email_body(msg['payload'])
                    
                    emails.append({
                        'id': email_id,
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'body': body,
                        'snippet': msg.get('snippet', ''),
                        'labels': msg.get('labelIds', [])
                    })
                    
                except Exception as e:
                    print(f"Error getting email {email_id}: {e}")
                    continue
            
            if emails:
                return self._process_emails_for_tasks(emails)
            
            return []
            
        except Exception as e:
            print(f"Error processing specific emails: {e}")
            return []

def create_realtime_processor(gmail_service: GmailService) -> GmailRealtimeProcessor:
    """
    Create a real-time email processor
    
    Args:
        gmail_service: Authenticated Gmail service
        
    Returns:
        GmailRealtimeProcessor instance
    """
    return GmailRealtimeProcessor(gmail_service) 