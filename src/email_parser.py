"""
Email Parser Module
Handles parsing of raw email files from the Enron dataset
"""

import os
import re
import email
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedEmail:
    """Structured representation of a parsed email"""
    message_id: str
    date: Optional[datetime]
    from_email: str
    to_emails: List[str]
    cc_emails: List[str]
    bcc_emails: List[str]
    subject: str
    body: str
    folder_path: str
    file_path: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'message_id': self.message_id,
            'date': self.date.isoformat() if self.date else None,
            'from_email': self.from_email,
            'to_emails': self.to_emails,
            'cc_emails': self.cc_emails,
            'bcc_emails': self.bcc_emails,
            'subject': self.subject,
            'body': self.body,
            'folder_path': self.folder_path,
            'file_path': self.file_path
        }


class EmailParser:
    """Parser for Enron email dataset"""
    
    def __init__(self, maildir_path: str = "maildir"):
        self.maildir_path = Path(maildir_path)
        
    def parse_email_file(self, file_path: Path) -> Optional[ParsedEmail]:
        """Parse a single email file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse email using email library
            msg = email.message_from_string(content)
            
            # Extract basic fields
            message_id = msg.get('Message-ID', '').strip('<>')
            subject = msg.get('Subject', '')
            from_email = self._extract_email(msg.get('From', ''))
            
            # Parse recipients
            to_emails = self._parse_recipients(msg.get('To', ''))
            cc_emails = self._parse_recipients(msg.get('Cc', ''))
            bcc_emails = self._parse_recipients(msg.get('Bcc', ''))
            
            # Parse date
            date_str = msg.get('Date', '')
            parsed_date = self._parse_date(date_str)
            
            # Extract body
            body = self._extract_body(msg)
            
            # Get folder info
            folder_path = str(file_path.parent.relative_to(self.maildir_path))
            
            return ParsedEmail(
                message_id=message_id,
                date=parsed_date,
                from_email=from_email,
                to_emails=to_emails,
                cc_emails=cc_emails,
                bcc_emails=bcc_emails,
                subject=subject,
                body=body,
                folder_path=folder_path,
                file_path=str(file_path)
            )
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_email(self, email_str: str) -> str:
        """Extract email address from string"""
        if not email_str:
            return ""
        
        # Look for email in angle brackets
        match = re.search(r'<([^>]+@[^>]+)>', email_str)
        if match:
            return match.group(1).lower()
        
        # Look for standalone email
        match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', email_str)
        if match:
            return match.group(1).lower()
        
        return email_str.strip().lower()
    
    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse comma-separated recipients"""
        if not recipients_str:
            return []
        
        recipients = []
        for recipient in recipients_str.split(','):
            email_addr = self._extract_email(recipient.strip())
            if email_addr and '@' in email_addr:
                recipients.append(email_addr)
        
        return recipients
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse email date string"""
        if not date_str:
            return None
        
        try:
            # Remove timezone info for simplicity
            date_str = re.sub(r'\s*\([^)]+\)$', '', date_str)
            date_str = re.sub(r'\s*[+-]\d{4}$', '', date_str)
            
            # Common email date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S',
                '%d %b %Y %H:%M:%S',
                '%a, %d %b %Y %H:%M',
                '%d %b %Y %H:%M',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _extract_body(self, msg) -> str:
        """Extract email body text"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode('utf-8', errors='ignore')
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('utf-8', errors='ignore')
        
        # Clean up body
        body = body.strip()
        return body
    
    def parse_all_emails(self, limit: Optional[int] = None) -> List[ParsedEmail]:
        """Parse all emails in the dataset"""
        emails = []
        count = 0
        
        for user_dir in self.maildir_path.iterdir():
            if not user_dir.is_dir():
                continue
                
            print(f"Processing user: {user_dir.name}")
            
            for folder in user_dir.rglob('*'):
                if folder.is_file():
                    # Check if it's an email file (numeric name with optional dot)
                    filename = folder.name
                    if filename.replace('.', '').isdigit():
                        parsed = self.parse_email_file(folder)
                        if parsed:
                            emails.append(parsed)
                            count += 1
                            
                            if limit and count >= limit:
                                return emails
                            
                            if count % 1000 == 0:
                                print(f"Parsed {count} emails...")
        
        print(f"Total emails parsed: {count}")
        return emails
    
    def get_user_emails(self, username: str) -> List[ParsedEmail]:
        """Get all emails for a specific user"""
        user_path = self.maildir_path / username
        if not user_path.exists():
            return []
        
        emails = []
        for folder in user_path.rglob('*'):
            if folder.is_file():
                # Check if it's an email file (numeric name with optional dot)
                filename = folder.name
                if filename.replace('.', '').isdigit():
                    parsed = self.parse_email_file(folder)
                    if parsed:
                        emails.append(parsed)
        
        return emails