"""
Google Calendar API Service
Handles authentication, calendar operations, and task synchronization
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import streamlit as st

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events'
]

class GoogleCalendarService:
    """Service class for Google Calendar operations"""
    
    def __init__(self):
        self.service = None
        self.credentials = None
        
    def authenticate(self) -> bool:
        """Authenticate with Google Calendar API"""
        try:
            # Check if credentials file exists
            creds = None
            if os.path.exists('token.json'):
                try:
                    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                except Exception as e:
                    st.warning(f"⚠️ Invalid token.json file: {str(e)}")
                    # Remove invalid token file
                    os.remove('token.json')
                    creds = None
            
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        st.warning(f"⚠️ Token refresh failed: {str(e)}")
                        # Remove invalid token file
                        if os.path.exists('token.json'):
                            os.remove('token.json')
                        creds = None
                
                if not creds:
                    # Check if credentials.json exists
                    if not os.path.exists('credentials.json'):
                        st.error("❌ Google Calendar credentials not found!")
                        st.info("""
                        **To use Google Calendar integration:**
                        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                        2. Create a new project or select existing one
                        3. Enable Google Calendar API
                        4. Create credentials (OAuth 2.0 Client ID)
                        5. Download `credentials.json` and place it in the project root
                        
                        **Common 500 Error Solutions:**
                        - Ensure Google Calendar API is enabled in your project
                        - Check that OAuth consent screen is configured
                        - Verify credentials.json is valid and complete
                        - Try deleting token.json and re-authenticating
                        """)
                        return False
                    
                    # Validate credentials.json
                    try:
                        with open('credentials.json', 'r') as f:
                            cred_data = json.load(f)
                        
                        if 'installed' not in cred_data and 'web' not in cred_data:
                            st.error("❌ Invalid credentials.json format!")
                            st.info("""
                            Your credentials.json should contain either 'installed' or 'web' configuration.
                            Please download a new OAuth 2.0 Client ID from Google Cloud Console.
                            """)
                            return False
                            
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON in credentials.json!")
                        return False
                    except Exception as e:
                        st.error(f"❌ Error reading credentials.json: {str(e)}")
                        return False
                    
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'credentials.json', SCOPES)
                        creds = flow.run_local_server(port=0)
                    except Exception as e:
                        st.error(f"❌ OAuth flow failed: {str(e)}")
                        st.info("""
                        **Troubleshooting OAuth Issues:**
                        1. Check your internet connection
                        2. Ensure credentials.json is valid
                        3. Try using a different browser
                        4. Check if your Google account has 2FA enabled
                        """)
                        return False
                
                # Save the credentials for the next run
                try:
                    with open('token.json', 'w') as token:
                        token.write(creds.to_json())
                except Exception as e:
                    st.warning(f"⚠️ Could not save token: {str(e)}")
            
            self.credentials = creds
            self.service = build('calendar', 'v3', credentials=creds)
            return True
            
        except Exception as e:
            st.error(f"❌ Google Calendar authentication failed: {str(e)}")
            st.info("""
            **Common Authentication Issues:**
            1. **500 Error**: Usually means API not enabled or credentials issue
            2. **Invalid Credentials**: Download new credentials.json
            3. **Network Issues**: Check internet connection
            4. **Permission Denied**: Ensure OAuth consent screen is configured
            
            **Quick Fixes:**
            - Delete token.json and try again
            - Re-download credentials.json from Google Cloud Console
            - Enable Google Calendar API in your project
            """)
            return False
    
    def get_calendars(self) -> List[Dict[str, Any]]:
        """Get list of available calendars"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []
            
            # Test API connection first
            try:
                calendar_list = self.service.calendarList().list().execute()
            except HttpError as error:
                if error.resp.status == 500:
                    st.error("❌ Google Calendar API 500 Error!")
                    st.info("""
                    **500 Error Solutions:**
                    1. **API Not Enabled**: Go to Google Cloud Console → APIs & Services → Library → Enable Google Calendar API
                    2. **Invalid Credentials**: Download new credentials.json
                    3. **OAuth Consent Screen**: Configure OAuth consent screen in Google Cloud Console
                    4. **Quota Exceeded**: Check API quotas in Google Cloud Console
                    5. **Service Account**: If using service account, ensure proper permissions
                    """)
                    return []
                else:
                    st.error(f"❌ Failed to get calendars: {error}")
                    return []
            
            calendars = calendar_list.get('items', [])
            
            if not calendars:
                st.warning("⚠️ No calendars found in your Google account")
                st.info("""
                **Possible Reasons:**
                1. No calendars created in Google Calendar
                2. Insufficient permissions
                3. Account not properly authenticated
                """)
            
            return [
                {
                    'id': cal['id'],
                    'summary': cal['summary'],
                    'primary': cal.get('primary', False),
                    'accessRole': cal.get('accessRole', 'none')
                }
                for cal in calendars
            ]
        except Exception as e:
            st.error(f"❌ Unexpected error getting calendars: {str(e)}")
            return []
    
    def get_events(self, calendar_id: str = 'primary', 
                   time_min: Optional[datetime] = None,
                   time_max: Optional[datetime] = None,
                   max_results: int = 100) -> List[Dict[str, Any]]:
        """Get events from specified calendar"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []
            
            # Default to current month if no dates specified
            if not time_min:
                time_min = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if not time_max:
                time_max = (time_min + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
            
            try:
                events_result = self.service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat() + 'Z',
                    timeMax=time_max.isoformat() + 'Z',
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
            except HttpError as error:
                if error.resp.status == 500:
                    st.error("❌ Google Calendar API 500 Error when fetching events!")
                    st.info("""
                    **Event Fetch 500 Error Solutions:**
                    1. **Calendar Access**: Ensure you have access to the selected calendar
                    2. **Date Range**: Try a smaller date range
                    3. **API Quota**: Check if you've exceeded API limits
                    4. **Calendar ID**: Verify the calendar ID is correct
                    """)
                    return []
                else:
                    st.error(f"❌ Failed to get events: {error}")
                    return []
            
            events = events_result.get('items', [])
            
            # Convert to standardized format
            formatted_events = []
            for event in events:
                try:
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    end = event['end'].get('dateTime', event['end'].get('date'))
                    
                    formatted_events.append({
                        'id': event['id'],
                        'title': event['summary'],
                        'description': event.get('description', ''),
                        'start': start,
                        'end': end,
                        'location': event.get('location', ''),
                        'attendees': [
                            attendee['email'] for attendee in event.get('attendees', [])
                        ],
                        'all_day': 'date' in event['start'],
                        'source': 'google_calendar'
                    })
                except Exception as e:
                    st.warning(f"⚠️ Skipping malformed event: {str(e)}")
                    continue
            
            return formatted_events
            
        except Exception as e:
            st.error(f"❌ Unexpected error getting events: {str(e)}")
            return []
    
    def create_event(self, calendar_id: str = 'primary', 
                    title: str = '', description: str = '',
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    all_day: bool = False,
                    location: str = '',
                    attendees: List[str] = None) -> Optional[str]:
        """Create a new calendar event"""
        try:
            if not self.service:
                if not self.authenticate():
                    return None
            
            event = {
                'summary': title,
                'description': description,
                'location': location,
            }
            
            if all_day:
                event['start'] = {
                    'date': start_time.strftime('%Y-%m-%d'),
                    'timeZone': 'UTC',
                }
                event['end'] = {
                    'date': end_time.strftime('%Y-%m-%d'),
                    'timeZone': 'UTC',
                }
            else:
                event['start'] = {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                }
                event['end'] = {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                }
            
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
            
            try:
                event = self.service.events().insert(
                    calendarId=calendar_id, body=event
                ).execute()
            except HttpError as error:
                if error.resp.status == 500:
                    st.error("❌ Google Calendar API 500 Error when creating event!")
                    st.info("""
                    **Event Creation 500 Error Solutions:**
                    1. **Calendar Permissions**: Ensure you have write access to the calendar
                    2. **Event Data**: Check that all required fields are valid
                    3. **Date Format**: Ensure dates are in correct format
                    4. **API Quota**: Check if you've exceeded API limits
                    """)
                    return None
                else:
                    st.error(f"❌ Failed to create event: {error}")
                    return None
            
            st.success(f"✅ Event '{title}' created successfully!")
            return event['id']
            
        except Exception as e:
            st.error(f"❌ Unexpected error creating event: {str(e)}")
            return None
    
    def update_event(self, calendar_id: str, event_id: str,
                    title: str = None, description: str = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    location: str = None) -> bool:
        """Update an existing calendar event"""
        try:
            if not self.service:
                if not self.authenticate():
                    return False
            
            # Get existing event
            try:
                event = self.service.events().get(
                    calendarId=calendar_id, eventId=event_id
                ).execute()
            except HttpError as error:
                if error.resp.status == 500:
                    st.error("❌ Google Calendar API 500 Error when fetching event!")
                    return False
                else:
                    st.error(f"❌ Failed to get event: {error}")
                    return False
            
            # Update fields
            if title:
                event['summary'] = title
            if description:
                event['description'] = description
            if location:
                event['location'] = location
            if start_time:
                if event['start'].get('date'):
                    event['start']['date'] = start_time.strftime('%Y-%m-%d')
                else:
                    event['start']['dateTime'] = start_time.isoformat()
            if end_time:
                if event['end'].get('date'):
                    event['end']['date'] = end_time.strftime('%Y-%m-%d')
                else:
                    event['end']['dateTime'] = end_time.isoformat()
            
            try:
                updated_event = self.service.events().update(
                    calendarId=calendar_id, eventId=event_id, body=event
                ).execute()
            except HttpError as error:
                if error.resp.status == 500:
                    st.error("❌ Google Calendar API 500 Error when updating event!")
                    return False
                else:
                    st.error(f"❌ Failed to update event: {error}")
                    return False
            
            st.success(f"✅ Event '{updated_event['summary']}' updated successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Unexpected error updating event: {str(e)}")
            return False
    
    def delete_event(self, calendar_id: str, event_id: str) -> bool:
        """Delete a calendar event"""
        try:
            if not self.service:
                if not self.authenticate():
                    return False
            
            try:
                self.service.events().delete(
                    calendarId=calendar_id, eventId=event_id
                ).execute()
            except HttpError as error:
                if error.resp.status == 500:
                    st.error("❌ Google Calendar API 500 Error when deleting event!")
                    return False
                else:
                    st.error(f"❌ Failed to delete event: {error}")
                    return False
            
            st.success("✅ Event deleted successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Unexpected error deleting event: {str(e)}")
            return False
    
    def sync_task_to_calendar(self, task_data: Dict[str, Any], 
                             calendar_id: str = 'primary') -> Optional[str]:
        """Sync a task from the system to Google Calendar"""
        try:
            task_name = task_data.get('task_name', 'Unnamed Task')
            summary = task_data.get('summary', '')
            start_date = task_data.get('start_date')
            due_date = task_data.get('due_date')
            
            # Create description with task details
            description = f"Task: {task_name}\n"
            if summary:
                description += f"Summary: {summary}\n"
            if task_data.get('owner_name'):
                description += f"Owner: {task_data['owner_name']}\n"
            if task_data.get('collaborators'):
                collab_names = [c.get('name', str(c)) if isinstance(c, dict) else str(c) 
                              for c in task_data['collaborators']]
                description += f"Collaborators: {', '.join(collab_names)}\n"
            
            # Use due_date as end, start_date as start, or create a day-long event
            if start_date and due_date:
                start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                all_day = False
            elif due_date:
                start_time = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                end_time = start_time + timedelta(hours=1)
                all_day = False
            elif start_date:
                start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_time = start_time + timedelta(hours=1)
                all_day = False
            else:
                # No dates, create for today
                start_time = datetime.now()
                end_time = start_time + timedelta(hours=1)
                all_day = False
            
            return self.create_event(
                calendar_id=calendar_id,
                title=task_name,
                description=description,
                start_time=start_time,
                end_time=end_time,
                all_day=all_day
            )
            
        except Exception as e:
            st.error(f"❌ Failed to sync task to calendar: {str(e)}")
            return None 