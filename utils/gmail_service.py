"""
Gmail API Service
Handles authentication, email operations, and task extraction from emails
"""

import os
import json
import base64
import email
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.labels'
]

class GmailService:
    """Service class for Gmail operations"""
    
    def __init__(self):
        self.service = None
        self.credentials = None
        
    def authenticate(self) -> bool:
        """Authenticate with Gmail API using environment variables"""
        try:
            # Check if credentials file exists
            creds = None
            if os.path.exists('gmail_token.json'):
                try:
                    creds = Credentials.from_authorized_user_file('gmail_token.json', SCOPES)
                except Exception as e:
                    st.warning(f"⚠️ Invalid gmail_token.json file: {str(e)}")
                    # Remove invalid token file
                    os.remove('gmail_token.json')
                    creds = None
            
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        st.warning(f"⚠️ Token refresh failed: {str(e)}")
                        # Remove invalid token file
                        if os.path.exists('gmail_token.json'):
                            os.remove('gmail_token.json')
                        creds = None
                
                if not creds:
                    # Check for credentials file
                    if not os.path.exists('credentials.json'):
                        st.error("❌ credentials.json not found!")
                        st.info("""
                        **To use Gmail integration:**
                        1. Download credentials.json from Google Cloud Console
                        2. Enable Gmail API in Google Cloud Console
                        3. Configure OAuth consent screen
                        """)
                        return False
                    
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                        # Use standard local server flow for non-Docker environments
                        creds = flow.run_local_server(port=0)
                    except Exception as e:
                        st.error(f"❌ OAuth flow failed: {str(e)}")
                        st.info("""
                        **Troubleshooting OAuth Issues:**
                        1. Check your internet connection
                        2. Ensure Gmail API is enabled in Google Cloud Console
                        3. Verify OAuth consent screen is configured
                        4. Check if your Google account has 2FA enabled
                        """)
                        return False
                
                # Save the credentials for the next run
                try:
                    with open('gmail_token.json', 'w') as token:
                        token.write(creds.to_json())
                except Exception as e:
                    st.warning(f"⚠️ Could not save token: {str(e)}")
            
            self.credentials = creds
            self.service = build('gmail', 'v1', credentials=creds)
            return True
            
        except Exception as e:
            st.error(f"❌ Gmail authentication failed: {str(e)}")
            return False
    
    def get_emails(self, query: str = '', max_results: int = 50) -> List[Dict[str, Any]]:
        """Get emails from Gmail inbox using proper Gmail API format"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []
            
            # Get email messages
            results = self.service.users().messages().list(
                userId='me', 
                q=query, 
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                # Get full message details with format='full'
                msg = self.service.users().messages().get(
                    userId='me', 
                    id=message['id'],
                    format='full'  # Get full message with all fields
                ).execute()
                
                # The message now contains all the Gmail API fields
                emails.append(msg)
            
            return emails
            
        except Exception as e:
            st.error(f"❌ Failed to get emails: {str(e)}")
            return []
    
    def _get_email_body(self, payload: Dict) -> str:
        """Extract email body from payload - now handled by the parser"""
        # This is now handled by the enhanced parser
        from gmail_parser.email_parser import extract_body_from_gmail_payload
        return extract_body_from_gmail_payload(payload)
    
    def send_email(self, to: str, subject: str, body: str, 
                   reply_to_message_id: Optional[str] = None) -> bool:
        """Send an email"""
        try:
            if not self.service:
                if not self.authenticate():
                    return False
            
            message = MIMEMultipart()
            message['to'] = to
            message['subject'] = subject
            
            msg = MIMEText(body)
            message.attach(msg)
            
            # If replying, add references
            if reply_to_message_id:
                message['In-Reply-To'] = reply_to_message_id
                message['References'] = reply_to_message_id
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to send email: {str(e)}")
            return False
    
    def get_labels(self) -> List[Dict[str, Any]]:
        """Get Gmail labels"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []
            
            results = self.service.users().labels().list(userId='me').execute()
            labels = results.get('labels', [])
            
            return [
                {
                    'id': label['id'],
                    'name': label['name'],
                    'type': label.get('type', 'user')
                }
                for label in labels
            ]
            
        except Exception as e:
            st.error(f"❌ Failed to get labels: {str(e)}")
            return []
    
    def create_label(self, name: str) -> Optional[str]:
        """Create a new Gmail label"""
        try:
            if not self.service:
                if not self.authenticate():
                    return None
            
            label_object = {
                'name': name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            
            result = self.service.users().labels().create(
                userId='me',
                body=label_object
            ).execute()
            
            return result['id']
            
        except Exception as e:
            st.error(f"❌ Failed to create label: {str(e)}")
            return None
    
    def add_label_to_message(self, message_id: str, label_id: str) -> bool:
        """Add a label to a message"""
        try:
            if not self.service:
                if not self.authenticate():
                    return False
            
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to add label: {str(e)}")
            return False
    
    def get_recent_emails(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get emails from the last N hours"""
        query = f'after:{(datetime.now() - timedelta(hours=hours)).strftime("%Y/%m/%d")}'
        return self.get_emails(query=query, max_results=100)
    
    def search_emails(self, search_query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search emails with Gmail search syntax"""
        return self.get_emails(query=search_query, max_results=max_results)
    
    def get_unread_emails(self, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get unread emails"""
        return self.get_emails(query='is:unread', max_results=max_results)
    
    def mark_as_read(self, message_id: str) -> bool:
        """Mark an email as read"""
        try:
            if not self.service:
                if not self.authenticate():
                    return False
            
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to mark as read: {str(e)}")
            return False
    
    def get_email_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all emails in a thread"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []
            
            thread = self.service.users().threads().get(
                userId='me',
                id=thread_id
            ).execute()
            
            emails = []
            for message in thread['messages']:
                headers = message['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                
                body = self._get_email_body(message['payload'])
                
                emails.append({
                    'id': message['id'],
                    'subject': subject,
                    'sender': sender,
                    'date': date,
                    'body': body,
                    'snippet': message.get('snippet', '')
                })
            
            return emails
            
        except Exception as e:
            st.error(f"❌ Failed to get thread: {str(e)}")
            return [] 