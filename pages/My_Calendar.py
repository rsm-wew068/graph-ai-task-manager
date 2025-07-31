import sys
import os

# Robust path fix for Hugging Face Spaces
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st
import pandas as pd
from streamlit_calendar import calendar
import datetime
import json

st.set_page_config(page_title="🗓 Calendar View", layout="wide")

# Custom CSS for Google Calendar-like styling with modal popup
st.markdown("""
<style>
    .task-event {
        border-radius: 4px !important;
        font-size: 12px !important;
        padding: 2px 4px !important;
        cursor: pointer !important;
    }
    
    /* Ensure all calendar events are clickable */
    .fc-event {
        cursor: pointer !important;
    }
    
    .fc-event:hover {
        opacity: 0.8 !important;
        transform: scale(1.02) !important;
        transition: all 0.2s ease !important;
    }
    
    .fc-event-title {
        font-weight: 500 !important;
    }
    
    .fc-daygrid-event {
        border-radius: 4px !important;
        margin: 1px !important;
    }
    
    .fc-toolbar-title {
        font-size: 1.5em !important;
        font-weight: 600 !important;
    }
    
    .fc-button {
        border-radius: 4px !important;
        font-size: 14px !important;
    }
    
    .stApp > div:first-child {
        padding-top: 1rem;
    }
    
    /* Modal styling for task popup */
    .task-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 9999;
        display: none; /* Initially hidden */
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(2px);
    }
    
    .task-modal-content {
        background: white;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        max-width: 500px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        position: relative;
        animation: modalSlideIn 0.3s ease-out;
        transform: scale(0.9);
        opacity: 0;
        transition: all 0.3s ease;
    }
    
    .task-modal[style*="flex"] .task-modal-content {
        transform: scale(1);
        opacity: 1;
    }
    
    @keyframes modalSlideIn {
        from {
            opacity: 0;
            transform: scale(0.9) translateY(-20px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    .task-modal-header {
        background: linear-gradient(135deg, #1976d2, #42a5f5);
        color: white;
        padding: 20px;
        border-radius: 12px 12px 0 0;
        position: relative;
    }
    
    .task-modal-close {
        position: absolute;
        top: 15px;
        right: 20px;
        background: none;
        border: none;
        color: white;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0.8;
        transition: all 0.2s ease;
    }
    
    .task-modal-close:hover {
        opacity: 1;
        background-color: rgba(255, 255, 255, 0.1);
        transform: scale(1.1);
    }
    
    .task-modal-body {
        padding: 24px;
    }
    
    .task-detail-section {
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .task-detail-section:last-child {
        border-bottom: none;
        margin-bottom: 0;
    }
    
    .task-detail-label {
        font-weight: 600;
        color: #666;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .task-detail-value {
        font-size: 14px;
        color: #333;
        line-height: 1.5;
    }
    
    .task-tag {
        display: inline-block;
        background: #f5f5f5;
        color: #666;
        padding: 4px 8px;
        border-radius: 16px;
        font-size: 12px;
        margin-right: 8px;
        margin-bottom: 4px;
    }
    
    .task-date-badge {
        background: linear-gradient(135deg, #4caf50, #66bb6a);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 8px;
    }
    
    .task-due-badge {
        background: linear-gradient(135deg, #ff9800, #ffb74d);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🗓 Task Calendar View")

# Initialize session state for modal and selected task
if "selected_task" not in st.session_state:
    st.session_state.selected_task = None
if "show_task_modal" not in st.session_state:
    st.session_state.show_task_modal = False
if "last_clicked_event" not in st.session_state:
    st.session_state.last_clicked_event = None

# Define functions first
def load_tasks_from_database():
    """Load tasks from PostgreSQL database"""
    try:
        from utils.database import PostgreSQLDatabase
        db = PostgreSQLDatabase()
        if db.connect():
            cursor = db.connection.cursor()
            cursor.execute("""
                SELECT task_name, task_description, topic, due_date, received_date, 
                       status, priority_level, sender, assigned_to, message_id, spam
                FROM tasks 
                WHERE due_date IS NOT NULL OR received_date IS NOT NULL
                ORDER BY COALESCE(due_date, received_date) DESC
            """)
            tasks = [dict(row) for row in cursor.fetchall()]
            db.close()
            return tasks
        else:
            return []
    except Exception as e:
        st.error(f"Error loading tasks from database: {e}")
        return []

def get_earliest_due_date(tasks):
    """Get the earliest due date from tasks"""
    earliest_date = None
    for task in tasks:
        due_date = task.get('due_date')
        if due_date:
            if earliest_date is None or due_date < earliest_date:
                earliest_date = due_date
    return earliest_date

def load_tasks_from_session():
    """Load and validate tasks from session state"""
    if (hasattr(st.session_state, 'processing_complete') and
            st.session_state.processing_complete):
        # Extract valid tasks from the outputs
        outputs = st.session_state.get("extracted_tasks", [])
        tasks = [
            res["validated_json"] for res in outputs
            if "validated_json" in res and res.get("valid", False)
        ]
        return tasks
    return []

# Load tasks first to get the earliest due date
session_tasks = load_tasks_from_session()
database_tasks = load_tasks_from_database()
tasks = database_tasks if database_tasks else session_tasks

# Get earliest due date for default
earliest_due_date = get_earliest_due_date(tasks)
if earliest_due_date:
    default_date = earliest_due_date
else:
    default_date = datetime.date(2006, 1, 1)  # Fallback

# Add date and month/year selection
st.markdown("---")
st.subheader("📅 Calendar Navigation")

col1, col2 = st.columns(2)
with col1:
    selected_month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=default_date.month - 1,
        format_func=lambda x: datetime.date(2000, x, 1).strftime("%B")
    )
with col2:
    selected_year = st.selectbox(
        "Year",
        options=list(range(1980, 2007)),
        index=default_date.year - 1980
    )

if st.button("📅 Go to Selected Month", type="primary"):
    st.session_state.calendar_start_date = datetime.date(selected_year, selected_month, 1)
    st.rerun()

# Display task information
if database_tasks:
    # Calculate date range for database tasks
    dates = []
    for task in database_tasks:
        if task.get('due_date'):
            dates.append(task['due_date'])
        if task.get('received_date'):
            dates.append(task['received_date'])
    
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        st.info(f"📊 Showing {len(database_tasks)} tasks from database (Date range: {min_date} to {max_date})")
    else:
        st.info(f"📊 Showing {len(database_tasks)} tasks from database")
elif session_tasks:
    st.info(f"📊 Showing {len(session_tasks)} tasks from current session")
else:
    st.warning("No tasks found in database or current session.")

if tasks is None:
    st.warning(
        "No tasks found. Please upload and process emails from the "
        "main page first."
    )
    st.info(
        "💡 Go to the main page to upload your Gmail Takeout ZIP file "
        "and process emails."
    )
else:
    st.subheader("📅 Task Calendar")
    st.markdown("*Click on any task to see detailed information*")

    # Convert tasks to event format
    events = []
    task_details = {}  # Store detailed task info for lookup
    
    for task_index, task_data in enumerate(tasks):
        try:
            # Handle both dict and string cases (same as flatten_extractions)
            if isinstance(task_data, str):
                try:
                    task_data = json.loads(task_data)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            # Skip non-dict items
            if not isinstance(task_data, dict):
                continue
            
            # Handle validated_json wrapper from HITL validation
            if "validated_json" in task_data:
                task_data = task_data["validated_json"]
            
            # Handle flat structure (new format) - PRIORITY
            if "task_name" in task_data:
                task_name = task_data.get("task_name", "Unnamed Task")
                task_description = task_data.get("task_description", "")
                topic = task_data.get("topic", "Unknown Topic")
                due_date = task_data.get("due_date")
                received_date = task_data.get("received_date")
                status = task_data.get("status", "not started")
                priority_level = task_data.get("priority_level", "Medium")
                sender = task_data.get("sender", "Unknown")
                assigned_to = task_data.get("assigned_to", "Unknown")
                message_id = task_data.get("message_id", "")
                spam = task_data.get("spam", False)
                
                # Use due_date or received_date for the calendar event
                event_date = due_date or received_date
                
                if event_date:  # Only add events with valid dates
                    # Convert date to ISO string format for JSON serialization
                    if hasattr(event_date, 'isoformat'):
                        event_date_str = event_date.isoformat()
                    else:
                        event_date_str = str(event_date)
                    
                    # Create unique event ID
                    event_id = f"task_{task_index}_{len(events)}"
                    
                    # Determine event type and styling
                    is_start_event = bool(received_date and not due_date)
                    # Blue for start, Orange for due
                    event_color = "#1976d2" if is_start_event else "#f57c00"
                
                    event = {
                        "id": event_id,
                        "title": task_name,
                        "start": event_date_str,
                        "allDay": True,
                        "backgroundColor": event_color,
                        "borderColor": event_color,
                        "textColor": "#ffffff"
                    }
                    events.append(event)
                    
                    # Store detailed info for this event
                    task_details[event_id] = {
                        "topic": topic,
                        "task_name": task_name,
                        "summary": task_description,
                        "sent_date": received_date,
                        "due_date": due_date,
                        "owner_name": sender,
                        "owner_role": "",
                        "owner_department": "",
                        "owner_organization": "",
                        "email_index": message_id,
                        "collaborators": [{"name": assigned_to, "role": "", "department": "", "organization": ""}] if assigned_to and assigned_to != "Unknown" else [],
                        "event_type": "Start" if is_start_event else "Due",
                        "status": status,
                        "priority_level": priority_level,
                        "spam": spam
                    }
                    
            # Handle old nested structure (for backward compatibility)
            elif "Topic" in task_data:
                topic = task_data.get("Topic", {})
                topic_name = (topic.get("name", "Unknown Topic")
                              if isinstance(topic, dict) else "Unknown Topic")
                topic_tasks = topic.get("tasks", [])
                
                for task_obj in topic_tasks:
                    if not isinstance(task_obj, dict):
                        continue
                        
                    task = task_obj.get("task", {})
                    if not isinstance(task, dict):
                        continue
                    
                    # Extract task details
                    task_name = task.get("name", "Unnamed Task")
                    sent_date = task.get("sent_date")
                    due_date = task.get("due_date")
                    summary = task.get("summary", "")
                    email_index = task_obj.get("email_index", "")
                    
                    # Handle owner
                    owner = task.get("owner", {})
                    if isinstance(owner, dict):
                        owner_name = owner.get("name", "Unknown")
                        owner_role = owner.get("role", "")
                        owner_dept = owner.get("department", "")
                        owner_org = owner.get("organization", "")
                    else:
                        owner_name = str(owner) if owner else "Unknown"
                        owner_role = owner_dept = owner_org = ""
                    
                    # Handle collaborators
                    collaborators = task.get("collaborators", [])
                    if not isinstance(collaborators, list):
                        collaborators = []
                    
                    # Use due_date or start_date for the calendar event
                    event_date = due_date or sent_date
                    
                    if event_date:  # Only add events with valid dates
                        # Convert date to ISO string format for JSON serialization
                        if hasattr(event_date, 'isoformat'):
                            event_date_str = event_date.isoformat()
                        else:
                            event_date_str = str(event_date)
                        
                        # Create unique event ID
                        event_id = f"task_{task_index}_{len(events)}"
                        
                        # Determine event type and styling
                        is_start_event = bool(sent_date)
                        # Blue for start, Orange for due
                        event_color = "#1976d2" if is_start_event else "#f57c00"
                    
                        event = {
                            "id": event_id,
                            "title": task_name,
                            "start": event_date_str,
                            "allDay": True,
                            "backgroundColor": event_color,
                            "borderColor": event_color,
                            "textColor": "#ffffff"
                        }
                        events.append(event)
                        
                        # Store detailed info for this event
                        task_details[event_id] = {
                            "topic": topic_name,
                            "task_name": task_name,
                            "summary": summary,
                            "sent_date": sent_date,
                            "due_date": due_date,
                            "owner_name": owner_name,
                            "owner_role": owner_role,
                            "owner_department": owner_dept,
                            "owner_organization": owner_org,
                            "email_index": email_index,
                            "collaborators": collaborators,
                            "event_type": "Start" if sent_date else "Due"
                        }
                    
        except Exception as e:
            st.error(f"Error processing task: {e}")
            continue

    # Set initial date based on session state or default to most recent task year
    if hasattr(st.session_state, 'calendar_start_date'):
        initial_date = st.session_state.calendar_start_date.strftime("%Y-%m-%d")
    else:
        # Default to 2006 (most recent tasks)
        initial_date = "2006-01-01"
    
    calendar_options = {
        "initialView": "dayGridMonth",
        "initialDate": initial_date,
        "editable": False,
        "selectable": True,
        "selectMirror": True,
        "dayMaxEvents": 3,
        "weekends": True,
        "navLinks": True,
        "headerToolbar": {
            "start": "prev,next",
            "center": "title",
            "end": "dayGridMonth,timeGridWeek,timeGridDay"
        },
        "height": 650,
        "businessHours": {
            "daysOfWeek": [1, 2, 3, 4, 5],
            "startTime": "08:00",
            "endTime": "18:00",
        }
    }

    # Show summary stats
    event_count = len(events)
    task_count = len(tasks)
    st.info(f"📊 Displaying {event_count} tasks from {task_count} emails")
    
    # Create full-width calendar - just like Google Calendar
    if events:
        # Render the calendar with unique key to prevent state persistence issues
        calendar_key = f"calendar_{event_count}_{hash(str(events))}"
        calendar_result = calendar(
            events=events,
            options=calendar_options,
            custom_css="""
            .fc-event {
                cursor: pointer !important;
            }
            .fc-event:hover {
                opacity: 0.8 !important;
                transform: scale(1.02) !important;
            }
            """,
            key=calendar_key
        )
        
        # Debug: Show what calendar_result contains
        if calendar_result and st.checkbox("🐛 Show Debug Info", key="debug_toggle"):
            st.write("Calendar result:")
            st.json(calendar_result)
        
        # Handle calendar interactions - Google Calendar style (no page refresh)
        if calendar_result.get("eventClick"):
            event_info = calendar_result["eventClick"]["event"]
            event_id = event_info.get("id")
            
            # Prevent duplicate processing of the same click
            if event_id != st.session_state.last_clicked_event:
                st.session_state.last_clicked_event = event_id
                
                if event_id and event_id in task_details:
                    st.session_state.selected_task = event_id
                    st.session_state.show_task_modal = True
                    # Remove st.rerun() - let Streamlit handle it naturally
                else:
                    st.error(f"Event ID {event_id} not found in task_details")
        
        # Show task details when a task is clicked
        if (st.session_state.show_task_modal and
            st.session_state.selected_task and
                st.session_state.selected_task in task_details):
            
            task_info = task_details[st.session_state.selected_task]
            
            # Comprehensive task details in expandable format
            with st.expander("📝 Task Details", expanded=True):
                st.markdown(f"**Task:** {task_info['task_name']}")
                st.markdown(f"**Topic:** {task_info['topic']}")
                
                # Status and Priority (new fields)
                if task_info.get('status'):
                    status_color = {
                        "not started": "🔴",
                        "in progress": "🟡", 
                        "completed": "🟢"
                    }.get(task_info['status'], "⚪")
                    st.markdown(f"• **Status:** {status_color} {task_info['status']}")
                
                if task_info.get('priority_level'):
                    priority_color = {
                        "High": "🔴",
                        "Medium": "🟡",
                        "Low": "🟢"
                    }.get(task_info['priority_level'], "⚪")
                    st.markdown(f"• **Priority:** {priority_color} {task_info['priority_level']}")
                
                # Spam indicator
                if task_info.get('spam'):
                    st.markdown(f"• **Spam:** 🚨 Yes")
                
                # Dates with bullet points
                if task_info['sent_date']:
                    st.markdown(f"• **Received Date:** {task_info['sent_date']}")
                if task_info['due_date']:
                    st.markdown(f"• **Due Date:** {task_info['due_date']}")
                
                # Description (flat structure)
                if task_info.get('summary'):
                    st.markdown(f"• **Description:** {task_info['summary']}")
                
                # Message ID (flat structure)
                if task_info.get('email_index'):
                    st.markdown(f"• **Message ID:** {task_info['email_index']}")
                
                # Sender (flat structure)
                if task_info.get('owner_name') and task_info['owner_name'] != 'Unknown':
                    st.markdown(f"• **Sender:** {task_info['owner_name']}")
                
                # Assigned To (flat structure)
                if task_info.get('collaborators'):
                    collab_list = []
                    for collab in task_info['collaborators']:
                        if isinstance(collab, dict):
                            collab_name = collab.get('name', 'Unknown')
                            if collab_name != 'Unknown':
                                collab_list.append(collab_name)
                        else:
                            collab_list.append(str(collab))
                    
                    if collab_list:
                        st.markdown(f"• **Assigned To:** {', '.join(collab_list)}")
                    else:
                        st.markdown(f"• **Assigned To:** Not specified")
                else:
                    st.markdown(f"• **Assigned To:** Not specified")
            
            # Simple close button
            if st.button("✖ Close Task Details", key="close_modal_simple",
                         type="primary", use_container_width=True):
                st.session_state.show_task_modal = False
                st.session_state.selected_task = None
                st.session_state.last_clicked_event = None
    else:
        st.warning("No tasks with valid dates found.")
        st.info("Tasks will appear here once you process emails with "
                "valid dates.")
