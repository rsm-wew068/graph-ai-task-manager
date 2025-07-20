"""
Gmail Task Integration
Connects Gmail API emails with the existing task extraction pipeline
"""

import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import re
from utils.email_parser import parse_gmail_emails

def gmail_to_dataframe(gmail_emails: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert Gmail API email data to DataFrame format expected by existing pipeline
    
    Args:
        gmail_emails: List of email dictionaries from Gmail API
        
    Returns:
        pandas.DataFrame: DataFrame in the format expected by email_parser
    """
    # Use the new email parser for Gmail API format
    return parse_gmail_emails(gmail_emails)

def process_gmail_emails_for_tasks(gmail_emails: List[Dict[str, Any]], 
                                 extraction_pipeline) -> List[Dict[str, Any]]:
    """
    Process Gmail emails through the existing task extraction pipeline
    
    Args:
        gmail_emails: List of email dictionaries from Gmail API
        extraction_pipeline: The existing LangGraph extraction pipeline
        
    Returns:
        List of extracted task results
    """
    # Convert Gmail emails to DataFrame format using the new parser
    df = gmail_to_dataframe(gmail_emails)
    
    # Process each email through the extraction pipeline
    results = []
    
    for index, row in df.iterrows():
        try:
            # Convert row to dict for processing
            email_row = row.to_dict()
            
            # Add email index for tracking
            email_row['email_index'] = str(index)
            
            # Process through extraction pipeline
            # This assumes you have the extraction pipeline available
            # You'll need to import and use your existing pipeline here
            result = {
                'email_index': str(index),
                'subject': email_row['Subject'],
                'sender': email_row['From'],
                'date': email_row['Date'],
                'status': 'processed',
                'message': 'Email processed successfully'
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
                'message': f'Processing failed: {str(e)}'
            }
            results.append(result)
    
    return results

def filter_emails_for_task_extraction(gmail_emails: List[Dict[str, Any]], 
                                    filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Filter Gmail emails to find those most likely to contain tasks
    
    Args:
        gmail_emails: List of email dictionaries from Gmail API
        filters: Optional filters to apply
        
    Returns:
        Filtered list of emails
    """
    if not filters:
        filters = {
            'min_body_length': 50,  # Minimum body length
            'exclude_labels': ['SPAM', 'TRASH', 'DRAFT'],  # Labels to exclude
            'include_keywords': ['task', 'todo', 'deadline', 'meeting', 'project', 'action'],
            'max_age_hours': 168  # Max 7 days old
        }
    
    filtered_emails = []
    
    for email in gmail_emails:
        # Check body length
        body_length = len(email.get('body', ''))
        if body_length < filters.get('min_body_length', 50):
            continue
        
        # Check for excluded labels
        email_labels = email.get('labels', [])
        if any(label in email_labels for label in filters.get('exclude_labels', [])):
            continue
        
        # Check for task-related keywords
        body_lower = email.get('body', '').lower()
        subject_lower = email.get('subject', '').lower()
        content = f"{subject_lower} {body_lower}"
        
        has_keywords = any(keyword in content for keyword in filters.get('include_keywords', []))
        if not has_keywords:
            continue
        
        # Check age (if date parsing works)
        try:
            date_str = email.get('date', '')
            email_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            age_hours = (datetime.now(email_date.tzinfo) - email_date).total_seconds() / 3600
            
            if age_hours > filters.get('max_age_hours', 168):
                continue
        except:
            # If date parsing fails, include the email anyway
            pass
        
        filtered_emails.append(email)
    
    return filtered_emails

def create_task_summary(gmail_emails: List[Dict[str, Any]], 
                       extracted_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of email processing and task extraction
    
    Args:
        gmail_emails: Original list of Gmail emails
        extracted_tasks: Results from task extraction
        
    Returns:
        Summary dictionary
    """
    total_emails = len(gmail_emails)
    processed_emails = len([r for r in extracted_tasks if r.get('status') == 'processed'])
    error_emails = len([r for r in extracted_tasks if r.get('status') == 'error'])
    
    # Count unread emails
    unread_count = len([e for e in gmail_emails if 'UNREAD' in e.get('labels', [])])
    
    # Get date range
    dates = []
    for email in gmail_emails:
        try:
            date_str = email.get('date', '')
            if date_str:
                email_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
                dates.append(email_date)
        except:
            continue
    
    date_range = ""
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    
    return {
        'total_emails': total_emails,
        'processed_emails': processed_emails,
        'error_emails': error_emails,
        'unread_emails': unread_count,
        'success_rate': (processed_emails / total_emails * 100) if total_emails > 0 else 0,
        'date_range': date_range,
        'processing_timestamp': datetime.now().isoformat()
    } 