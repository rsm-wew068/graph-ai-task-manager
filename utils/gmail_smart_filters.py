"""
Smart Gmail Email Filtering
Advanced filtering to identify emails most likely to contain tasks
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EmailFilter:
    """Email filter configuration"""
    min_body_length: int = 50
    max_body_length: int = 10000  # Avoid very long emails
    exclude_labels: List[str] = None
    include_labels: List[str] = None
    exclude_senders: List[str] = None
    include_senders: List[str] = None
    task_keywords: List[str] = None
    urgency_keywords: List[str] = None
    meeting_keywords: List[str] = None
    project_keywords: List[str] = None
    max_age_hours: int = 168  # 7 days
    min_age_hours: int = 0
    require_attachments: bool = False
    exclude_auto_replies: bool = True
    exclude_newsletters: bool = True

class GmailSmartFilter:
    """Advanced email filtering for task extraction"""
    
    def __init__(self, filter_config: Optional[EmailFilter] = None):
        self.config = filter_config or EmailFilter()
        self._setup_default_keywords()
    
    def _setup_default_keywords(self):
        """Setup default keyword lists if not provided"""
        if not self.config.task_keywords:
            self.config.task_keywords = [
                'task', 'todo', 'to-do', 'action item', 'action required',
                'deadline', 'due date', 'urgent', 'priority', 'follow up',
                'review', 'approve', 'sign', 'complete', 'finish',
                'submit', 'send', 'prepare', 'create', 'update'
            ]
        
        if not self.config.urgency_keywords:
            self.config.urgency_keywords = [
                'urgent', 'asap', 'immediately', 'critical', 'emergency',
                'deadline', 'due today', 'due tomorrow', 'priority',
                'important', 'high priority', 'rush'
            ]
        
        if not self.config.meeting_keywords:
            self.config.meeting_keywords = [
                'meeting', 'call', 'conference', 'discussion', 'sync',
                'standup', 'review', 'presentation', 'demo', 'workshop',
                'brainstorm', 'planning', 'sprint'
            ]
        
        if not self.config.project_keywords:
            self.config.project_keywords = [
                'project', 'initiative', 'campaign', 'launch', 'release',
                'milestone', 'phase', 'sprint', 'iteration', 'deliverable'
            ]
        
        if not self.config.exclude_labels:
            self.config.exclude_labels = ['SPAM', 'TRASH', 'DRAFT', 'SENT']
        
        if not self.config.exclude_senders:
            self.config.exclude_senders = [
                'noreply', 'no-reply', 'donotreply', 'do-not-reply',
                'automated', 'system', 'notification'
            ]
    
    def filter_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply smart filtering to emails
        
        Args:
            emails: List of Gmail email dictionaries
            
        Returns:
            Filtered list of emails
        """
        filtered_emails = []
        
        for email in emails:
            if self._should_include_email(email):
                # Add filter scores for ranking
                email['filter_score'] = self._calculate_filter_score(email)
                filtered_emails.append(email)
        
        # Sort by filter score (highest first)
        filtered_emails.sort(key=lambda x: x.get('filter_score', 0), reverse=True)
        
        return filtered_emails
    
    def _should_include_email(self, email: Dict[str, Any]) -> bool:
        """Check if email should be included based on filters"""
        
        # Check body length
        body_length = len(email.get('body', ''))
        if body_length < self.config.min_body_length:
            return False
        if body_length > self.config.max_body_length:
            return False
        
        # Check excluded labels
        email_labels = email.get('labels', [])
        if any(label in email_labels for label in self.config.exclude_labels):
            return False
        
        # Check included labels (if specified)
        if self.config.include_labels:
            if not any(label in email_labels for label in self.config.include_labels):
                return False
        
        # Check sender filters
        sender = email.get('sender', '').lower()
        if any(excluded in sender for excluded in self.config.exclude_senders):
            return False
        
        if self.config.include_senders:
            if not any(included in sender for included in self.config.include_senders):
                return False
        
        # Check age
        if not self._is_within_age_range(email):
            return False
        
        # Check for auto-replies
        if self.config.exclude_auto_replies and self._is_auto_reply(email):
            return False
        
        # Check for newsletters
        if self.config.exclude_newsletters and self._is_newsletter(email):
            return False
        
        # Check for task-related content
        if not self._has_task_content(email):
            return False
        
        return True
    
    def _is_within_age_range(self, email: Dict[str, Any]) -> bool:
        """Check if email is within the specified age range"""
        try:
            date_str = email.get('date', '')
            email_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            now = datetime.now(email_date.tzinfo)
            age_hours = (now - email_date).total_seconds() / 3600
            
            return (self.config.min_age_hours <= age_hours <= self.config.max_age_hours)
        except:
            # If date parsing fails, include the email
            return True
    
    def _is_auto_reply(self, email: Dict[str, Any]) -> bool:
        """Check if email is an auto-reply"""
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        
        auto_reply_indicators = [
            'out of office', 'automatic reply', 'auto reply', 'auto-reply',
            'vacation', 'away', 'unavailable', 'out of the office',
            'automatic response', 'auto response'
        ]
        
        return any(indicator in subject or indicator in body 
                  for indicator in auto_reply_indicators)
    
    def _is_newsletter(self, email: Dict[str, Any]) -> bool:
        """Check if email is a newsletter"""
        subject = email.get('subject', '').lower()
        sender = email.get('sender', '').lower()
        
        newsletter_indicators = [
            'newsletter', 'digest', 'weekly', 'monthly', 'roundup',
            'summary', 'update', 'news', 'announcement'
        ]
        
        return any(indicator in subject or indicator in sender 
                  for indicator in newsletter_indicators)
    
    def _has_task_content(self, email: Dict[str, Any]) -> bool:
        """Check if email contains task-related content"""
        content = f"{email.get('subject', '')} {email.get('body', '')}".lower()
        
        # Check for task keywords
        has_task_keywords = any(keyword in content for keyword in self.config.task_keywords)
        
        # Check for urgency keywords
        has_urgency_keywords = any(keyword in content for keyword in self.config.urgency_keywords)
        
        # Check for meeting keywords
        has_meeting_keywords = any(keyword in content for keyword in self.config.meeting_keywords)
        
        # Check for project keywords
        has_project_keywords = any(keyword in content for keyword in self.config.project_keywords)
        
        return has_task_keywords or has_urgency_keywords or has_meeting_keywords or has_project_keywords
    
    def _calculate_filter_score(self, email: Dict[str, Any]) -> float:
        """Calculate a score for email prioritization"""
        score = 0.0
        content = f"{email.get('subject', '')} {email.get('body', '')}".lower()
        
        # Base score for being unread
        if 'UNREAD' in email.get('labels', []):
            score += 10.0
        
        # Score for task keywords
        task_matches = sum(1 for keyword in self.config.task_keywords if keyword in content)
        score += task_matches * 5.0
        
        # Score for urgency keywords
        urgency_matches = sum(1 for keyword in self.config.urgency_keywords if keyword in content)
        score += urgency_matches * 8.0
        
        # Score for meeting keywords
        meeting_matches = sum(1 for keyword in self.config.meeting_keywords if keyword in content)
        score += meeting_matches * 6.0
        
        # Score for project keywords
        project_matches = sum(1 for keyword in self.config.project_keywords if keyword in content)
        score += project_matches * 4.0
        
        # Score for recent emails (within last 24 hours)
        try:
            date_str = email.get('date', '')
            email_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            now = datetime.now(email_date.tzinfo)
            age_hours = (now - email_date).total_seconds() / 3600
            
            if age_hours <= 24:
                score += 15.0
            elif age_hours <= 72:
                score += 10.0
            elif age_hours <= 168:
                score += 5.0
        except:
            pass
        
        # Score for important labels
        important_labels = ['IMPORTANT', 'STARRED', 'INBOX']
        email_labels = email.get('labels', [])
        for label in important_labels:
            if label in email_labels:
                score += 3.0
        
        return score

def create_task_focused_filter() -> GmailSmartFilter:
    """Create a filter optimized for task extraction"""
    config = EmailFilter(
        min_body_length=30,
        max_body_length=5000,
        task_keywords=[
            'task', 'todo', 'action item', 'deadline', 'due date',
            'urgent', 'priority', 'follow up', 'review', 'approve'
        ],
        urgency_keywords=[
            'urgent', 'asap', 'immediately', 'critical', 'deadline'
        ],
        max_age_hours=336,  # 14 days
        exclude_auto_replies=True,
        exclude_newsletters=True
    )
    return GmailSmartFilter(config)

def create_meeting_focused_filter() -> GmailSmartFilter:
    """Create a filter optimized for meeting extraction"""
    config = EmailFilter(
        min_body_length=20,
        max_body_length=3000,
        meeting_keywords=[
            'meeting', 'call', 'conference', 'discussion', 'sync',
            'standup', 'presentation', 'demo', 'workshop'
        ],
        max_age_hours=168,  # 7 days
        exclude_auto_replies=True
    )
    return GmailSmartFilter(config)

def create_project_focused_filter() -> GmailSmartFilter:
    """Create a filter optimized for project-related emails"""
    config = EmailFilter(
        min_body_length=50,
        max_body_length=8000,
        project_keywords=[
            'project', 'initiative', 'campaign', 'launch', 'release',
            'milestone', 'phase', 'sprint', 'deliverable'
        ],
        max_age_hours=720,  # 30 days
        exclude_auto_replies=True
    )
    return GmailSmartFilter(config) 

def create_user_settings_filter(settings) -> GmailSmartFilter:
    """
    Create a GmailSmartFilter using user-provided settings from session state.
    Args:
        settings: dict with keys 'date_range', 'min_length', 'max_length', 'keywords', 'email_types'
    Returns:
        GmailSmartFilter instance
    """
    from datetime import datetime
    # Calculate max_age_hours and min_age_hours from date_range
    now = datetime.now()
    start_date, end_date = settings.get('date_range', (now, now))
    # Ensure start_date and end_date are datetime objects
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)
    min_age_hours = max(0, (now - end_date).total_seconds() / 3600)
    max_age_hours = max(1, (now - start_date).total_seconds() / 3600)
    # Map email_types to Gmail categories/labels
    include_labels = []
    type_to_label = {
        'Primary': 'CATEGORY_PERSONAL',
        'Promotions': 'CATEGORY_PROMOTIONS',
        'Social': 'CATEGORY_SOCIAL',
        'Updates': 'CATEGORY_UPDATES',
    }
    for t in settings.get('email_types', []):
        if t in type_to_label:
            include_labels.append(type_to_label[t])
    config = EmailFilter(
        min_body_length=settings.get('min_length', 50),
        max_body_length=settings.get('max_length', 5000),
        task_keywords=settings.get('keywords', []),
        min_age_hours=min_age_hours,
        max_age_hours=max_age_hours,
        include_labels=include_labels if include_labels else None,
        exclude_auto_replies=True,
        exclude_newsletters=True
    )
    return GmailSmartFilter(config) 