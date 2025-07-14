import pandas as pd
import plotly.express as px
from ics import Calendar, Event
from datetime import datetime
import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar']

def convert_graph_tasks_to_list(G):
    """Extracts all tasks from graph into a list of dictionaries."""
    tasks = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "task" or data.get("label", "").lower() == "task":
            tasks.append({
                "id": node_id,
                "name": data.get("name", "Untitled Task"),
                "summary": data.get("summary", ""),
                "start_date": data.get("start_date"),
                "due_date": data.get("due_date")
            })
    return tasks

def generate_gantt_chart(tasks):
    if not tasks:
        print("⚠️ No tasks passed to Gantt chart")
        return None
    df = pd.DataFrame(tasks)
    fig = px.timeline(df, x_start="start_date", x_end="due_date", y="name", title="Scheduled Tasks")
    fig.update_yaxes(autorange="reversed")
    return fig

def export_to_ics(tasks):
    """Generate an iCalendar (.ics) string from tasks."""
    cal = Calendar()
    for task in tasks:
        e = Event()
        e.name = task.get("name")
        e.begin = task.get("start_date")
        e.end = task.get("due_date")
        e.description = task.get("summary", "")
        cal.events.add(e)
    return cal.serialize()

def setup_google_calendar_auth():
    """
    Set up Google Calendar API authentication.
    Returns the authenticated service object.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError(
                    "credentials.json not found. Please download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('calendar', 'v3', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def create_google_calendar_event(service, task, calendar_id='primary'):
    """
    Create a single event in Google Calendar from a task.
    """
    try:
        # Convert datetime objects to RFC3339 format if needed
        start_time = task.get('start_date')
        end_time = task.get('due_date')
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        event = {
            'summary': task.get('name', 'Untitled Task'),
            'description': task.get('summary', ''),
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'UTC',
            },
        }
        
        event = service.events().insert(calendarId=calendar_id, body=event).execute()
        print(f'Event created: {event.get("htmlLink")}')
        return event
    
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def export_tasks_to_google_calendar(tasks, calendar_id='primary'):
    """
    Export selected tasks to Google Calendar.
    Each task must include: name, summary, start_date, due_date
    """
    service = setup_google_calendar_auth()
    if not service:
        print("❌ Failed to authenticate with Google Calendar")
        return False
    
    created_events = []
    for task in tasks:
        if not task.get("start_date") or not task.get("due_date"):
            print(f"⚠️ Skipping task '{task.get('name', 'Unnamed Task')}' (missing dates)")
            continue

        event = create_google_calendar_event(service, task, calendar_id)
        if event:
            print(f"✅ Scheduled task: {task['name']}")
            created_events.append(event)
        else:
            print(f"❌ Failed to schedule task: {task['name']}")
    
    print(f"\n📅 Successfully created {len(created_events)} events in Google Calendar")
    return created_events

def list_google_calendars():
    """
    List all available Google Calendars for the authenticated user.
    """
    service = setup_google_calendar_auth()
    if not service:
        return []
    
    try:
        calendar_list = service.calendarList().list().execute()
        calendars = calendar_list.get('items', [])
        
        print("Available calendars:")
        for calendar in calendars:
            print(f"- {calendar['summary']} (ID: {calendar['id']})")
        
        return calendars
    
    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

def schedule_calendar_tasks(G, input, calendar_id='primary'):
    """
    Schedule selected tasks from the graph to Google Calendar.
    """
    selected_task_ids = input.task_select()
    tasks_to_schedule = []
    for node_id in selected_task_ids:
        data = G.nodes.get(node_id, {})
        task = {
            "name": data.get("name", "Untitled Task"),
            "summary": data.get("summary", ""),
            "start_date": data.get("start_date"),
            "due_date": data.get("due_date")
        }
        tasks_to_schedule.append(task)
    export_tasks_to_google_calendar(tasks_to_schedule, calendar_id)