import streamlit as st
from streamlit_calendar import calendar
import json
import sys
import os

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from google_calendar_service import GoogleCalendarService
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False

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
    
    /* Google Calendar iframe styling */
    .google-calendar-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Calendar type selector styling */
    .calendar-type-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    /* Event source badges */
    .event-source-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 500;
        margin-left: 4px;
    }
    
    .event-source-task {
        background: #e3f2fd;
        color: #1976d2;
    }
    
    .event-source-google {
        background: #f3e5f5;
        color: #7b1fa2;
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

st.title("🗓 Unified Calendar View")

# Initialize session state
if "selected_task" not in st.session_state:
    st.session_state.selected_task = None
if "show_task_modal" not in st.session_state:
    st.session_state.show_task_modal = False
if "last_clicked_event" not in st.session_state:
    st.session_state.last_clicked_event = None
if "calendar_type" not in st.session_state:
    st.session_state.calendar_type = "unified"
if "google_calendar_service" not in st.session_state:
    st.session_state.google_calendar_service = None

# Calendar type selector
st.markdown("### 📅 Calendar Type")
calendar_type = st.selectbox(
    "Choose your calendar view:",
    options=[
        ("unified", "🔄 Unified View (Tasks + Google Calendar)"),
        ("tasks", "📋 Task Calendar Only"),
        ("google", "📅 Google Calendar Only")
    ],
    format_func=lambda x: x[1],
    key="calendar_type_selector"
)
st.session_state.calendar_type = calendar_type[0]

# Google Calendar setup
if GOOGLE_CALENDAR_AVAILABLE and st.session_state.calendar_type in ["unified", "google"]:
    st.markdown("### 🔗 Google Calendar Integration")
    
    # Initialize Google Calendar service
    if st.session_state.google_calendar_service is None:
        st.session_state.google_calendar_service = GoogleCalendarService()
    
    # Check if credentials exist
    credentials_exist = os.path.exists('credentials.json')
    token_exists = os.path.exists('token.json')
    
    if not credentials_exist:
        st.warning("⚠️ Google Calendar credentials not found!")
        st.info("""
        **To enable Google Calendar integration:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing one
        3. Enable Google Calendar API
        4. Create credentials (OAuth 2.0 Client ID)
        5. Download `credentials.json` and place it in the project root
        """)
        
        # Show embedded Google Calendar as fallback
        st.markdown("### 📅 Google Calendar (Embedded)")
        st.info("🔗 You can still view your Google Calendar by embedding it directly:")
        
        col1, col2 = st.columns(2)
        with col1:
            calendar_url = st.text_input(
                "Enter your Google Calendar embed URL:",
                placeholder="https://calendar.google.com/calendar/embed?src=..."
            )
        with col2:
            if st.button("🔗 Embed Calendar", disabled=not calendar_url):
                st.session_state.embedded_calendar_url = calendar_url
        
        if hasattr(st.session_state, 'embedded_calendar_url') and st.session_state.embedded_calendar_url:
            st.markdown("### 📅 Your Google Calendar")
            st.components.v1.iframe(
                st.session_state.embedded_calendar_url,
                height=600,
                scrolling=True
            )
    
    else:
        # Authenticate with Google Calendar
        if st.button("🔐 Connect to Google Calendar"):
            with st.spinner("Connecting to Google Calendar..."):
                if st.session_state.google_calendar_service.authenticate():
                    st.success("✅ Connected to Google Calendar!")
                    st.rerun()
                else:
                    st.error("❌ Failed to connect to Google Calendar")
        
        # Show Google Calendar options if authenticated
        if token_exists:
            try:
                if st.session_state.google_calendar_service.authenticate():
                    calendars = st.session_state.google_calendar_service.get_calendars()
                    
                    if calendars:
                        st.success("✅ Google Calendar connected!")
                        
                        # Calendar selection
                        calendar_options = {cal['summary']: cal['id'] for cal in calendars}
                        selected_calendar = st.selectbox(
                            "Select Google Calendar:",
                            options=list(calendar_options.keys()),
                            index=0
                        )
                        
                        # Sync options
                        st.markdown("#### 🔄 Sync Options")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📥 Import Google Calendar Events"):
                                with st.spinner("Importing events..."):
                                    events = st.session_state.google_calendar_service.get_events(
                                        calendar_id=calendar_options[selected_calendar]
                                    )
                                    if events:
                                        st.session_state.google_events = events
                                        st.success(f"✅ Imported {len(events)} events!")
                                    else:
                                        st.warning("No events found in the selected time range")
                        
                        with col2:
                            if st.button("📤 Sync Tasks to Google Calendar"):
                                # This will be implemented in the task processing section
                                st.info("This feature will sync your tasks to Google Calendar")
                        
                        # Show Google Calendar iframe
                        if st.session_state.calendar_type == "google":
                            st.markdown("### 📅 Google Calendar")
                            calendar_embed_url = f"https://calendar.google.com/calendar/embed?src={calendar_options[selected_calendar]}"
                            st.components.v1.iframe(
                                calendar_embed_url,
                                height=700,
                                scrolling=True
                            )
                    
                    else:
                        st.warning("No calendars found in your Google account")
                        
            except Exception as e:
                st.error(f"❌ Google Calendar error: {str(e)}")

# Load tasks from session
def load_tasks_from_session():
    """Load and validate tasks from session state"""
    if (hasattr(st.session_state, 'processing_complete') and
            st.session_state.processing_complete):
        outputs = st.session_state.get("extracted_tasks", [])
        tasks = [
            res["validated_json"] for res in outputs
            if "validated_json" in res and res.get("valid", False)
        ]
        return tasks
    return None

# Process tasks and create events
tasks = load_tasks_from_session()
events = []
task_details = {}

if tasks and st.session_state.calendar_type in ["unified", "tasks"]:
    st.markdown("### 📋 Task Processing")
    
    for task_index, task_data in enumerate(tasks):
        try:
            if isinstance(task_data, str):
                try:
                    task_data = json.loads(task_data)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if not isinstance(task_data, dict):
                continue
                
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
                
                task_name = task.get("name", "Unnamed Task")
                start_date = task.get("start_date")
                due_date = task.get("due_date")
                summary = task.get("summary", "")
                email_index = task_obj.get("email_index", "")
                
                owner = task.get("owner", {})
                if isinstance(owner, dict):
                    owner_name = owner.get("name", "Unknown")
                    owner_role = owner.get("role", "")
                    owner_dept = owner.get("department", "")
                    owner_org = owner.get("organization", "")
                else:
                    owner_name = str(owner) if owner else "Unknown"
                    owner_role = owner_dept = owner_org = ""
                
                collaborators = task.get("collaborators", [])
                if not isinstance(collaborators, list):
                    collaborators = []
                
                event_date = due_date or start_date
                
                if event_date:
                    event_id = f"task_{task_index}_{len(events)}"
                    is_start_event = bool(start_date)
                    event_color = "#1976d2" if is_start_event else "#f57c00"
                
                    event = {
                        "id": event_id,
                        "title": f"{task_name} 📋",
                        "start": event_date,
                        "allDay": True,
                        "backgroundColor": event_color,
                        "borderColor": event_color,
                        "textColor": "#ffffff"
                    }
                    events.append(event)
                    
                    task_details[event_id] = {
                        "topic": topic_name,
                        "task_name": task_name,
                        "summary": summary,
                        "start_date": start_date,
                        "due_date": due_date,
                        "owner_name": owner_name,
                        "owner_role": owner_role,
                        "owner_department": owner_dept,
                        "owner_organization": owner_org,
                        "email_index": email_index,
                        "collaborators": collaborators,
                        "event_type": "Start" if start_date else "Due",
                        "source": "task"
                    }
                    
        except Exception as e:
            st.error(f"Error processing task: {e}")
            continue

# Add Google Calendar events if available
if (GOOGLE_CALENDAR_AVAILABLE and 
    st.session_state.calendar_type in ["unified", "google"] and
    hasattr(st.session_state, 'google_events')):
    
    for event in st.session_state.google_events:
        event_id = f"google_{event['id']}"
        events.append({
            "id": event_id,
            "title": f"{event['title']} 📅",
            "start": event['start'],
            "end": event['end'],
            "allDay": event.get('all_day', False),
            "backgroundColor": "#7b1fa2",
            "borderColor": "#7b1fa2",
            "textColor": "#ffffff"
        })
        
        task_details[event_id] = {
            "task_name": event['title'],
            "summary": event.get('description', ''),
            "start_date": event['start'],
            "end_date": event['end'],
            "location": event.get('location', ''),
            "attendees": event.get('attendees', []),
            "source": "google_calendar"
        }

# Display calendar
if st.session_state.calendar_type == "unified":
    st.markdown("### 🔄 Unified Calendar View")
    st.info("📊 Shows both your tasks and Google Calendar events")
elif st.session_state.calendar_type == "tasks":
    st.markdown("### 📋 Task Calendar View")
    st.info("📊 Shows only your extracted tasks")
elif st.session_state.calendar_type == "google":
    st.markdown("### 📅 Google Calendar View")
    st.info("📊 Shows only your Google Calendar events")

if events:
    # Show summary stats
    task_events = [e for e in events if e['id'].startswith('task_')]
    google_events = [e for e in events if e['id'].startswith('google_')]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", len(events))
    with col2:
        st.metric("Task Events", len(task_events))
    with col3:
        st.metric("Google Events", len(google_events))
    
    # Calendar options
    calendar_options = {
        "initialView": "dayGridMonth",
        "editable": False,
        "selectable": True,
        "selectMirror": True,
        "dayMaxEvents": 5,
        "weekends": True,
        "navLinks": True,
        "headerToolbar": {
            "start": "prev,next today",
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
    
    # Render calendar
    calendar_key = f"calendar_{len(events)}_{hash(str(events))}"
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
    
    # Handle calendar interactions
    if calendar_result.get("eventClick"):
        event_info = calendar_result["eventClick"]["event"]
        event_id = event_info.get("id")
        
        if event_id != st.session_state.last_clicked_event:
            st.session_state.last_clicked_event = event_id
            
            if event_id and event_id in task_details:
                st.session_state.selected_task = event_id
                st.session_state.show_task_modal = True
    
    # Show event details
    if (st.session_state.show_task_modal and
        st.session_state.selected_task and
        st.session_state.selected_task in task_details):
        
        event_info = task_details[st.session_state.selected_task]
        
        with st.expander("📝 Event Details", expanded=True):
            st.markdown(f"**Event:** {event_info['task_name']}")
            
            # Show source badge
            source = event_info.get('source', 'unknown')
            if source == 'task':
                st.markdown('<span class="event-source-badge event-source-task">📋 Task</span>', unsafe_allow_html=True)
            elif source == 'google_calendar':
                st.markdown('<span class="event-source-badge event-source-google">📅 Google Calendar</span>', unsafe_allow_html=True)
            
            # Task-specific details
            if source == 'task':
                st.markdown(f"**Topic:** {event_info.get('topic', 'N/A')}")
                
                if event_info.get('start_date'):
                    st.markdown(f"• **Start Date:** {event_info['start_date']}")
                if event_info.get('due_date'):
                    st.markdown(f"• **Due Date:** {event_info['due_date']}")
                
                if event_info.get('summary'):
                    st.markdown(f"• **Summary:** {event_info['summary']}")
                
                if event_info.get('email_index'):
                    st.markdown(f"• **Email Index:** {event_info['email_index']}")
                
                # Owner details
                owner_name = event_info.get('owner_name', 'Unknown')
                owner_role = event_info.get('owner_role', 'Unknown')
                owner_dept = event_info.get('owner_department', 'Unknown') 
                owner_org = event_info.get('owner_organization', 'Unknown')
                
                responsible_to = f"{owner_name}"
                if owner_role != 'Unknown' or owner_dept != 'Unknown' or owner_org != 'Unknown':
                    role_details = []
                    if owner_role != 'Unknown':
                        role_details.append(f"Role: {owner_role}")
                    if owner_dept != 'Unknown':
                        role_details.append(f"Department: {owner_dept}")
                    if owner_org != 'Unknown':
                        role_details.append(f"Organization: {owner_org}")
                    
                    if role_details:
                        responsible_to += f" ({', '.join(role_details)})"
                
                st.markdown(f"• **Responsible To:** {responsible_to}")
                
                # Collaborators
                if event_info.get('collaborators'):
                    collab_list = []
                    for collab in event_info['collaborators']:
                        if isinstance(collab, dict):
                            collab_name = collab.get('name', 'Unknown')
                            collab_role = collab.get('role', 'Unknown')
                            collab_dept = collab.get('department', 'Unknown')
                            collab_org = collab.get('organization', 'Unknown')
                            
                            collab_details = f"{collab_name}"
                            if collab_role != 'Unknown' or collab_dept != 'Unknown' or collab_org != 'Unknown':
                                role_parts = []
                                if collab_role != 'Unknown':
                                    role_parts.append(f"Role: {collab_role}")
                                if collab_dept != 'Unknown':
                                    role_parts.append(f"Department: {collab_dept}")
                                if collab_org != 'Unknown':
                                    role_parts.append(f"Organization: {collab_org}")
                                
                                if role_parts:
                                    collab_details += f" ({', '.join(role_parts)})"
                            
                            collab_list.append(collab_details)
                        else:
                            collab_list.append(str(collab))
                    
                    if collab_list:
                        st.markdown(f"• **Collaborated By:** {', '.join(collab_list)}")
                    else:
                        st.markdown(f"• **Collaborated By:** None")
                else:
                    st.markdown(f"• **Collaborated By:** None")
            
            # Google Calendar event details
            elif source == 'google_calendar':
                if event_info.get('start_date'):
                    st.markdown(f"• **Start:** {event_info['start_date']}")
                if event_info.get('end_date'):
                    st.markdown(f"• **End:** {event_info['end_date']}")
                if event_info.get('location'):
                    st.markdown(f"• **Location:** {event_info['location']}")
                if event_info.get('summary'):
                    st.markdown(f"• **Description:** {event_info['summary']}")
                if event_info.get('attendees'):
                    st.markdown(f"• **Attendees:** {', '.join(event_info['attendees'])}")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✖ Close", key="close_event_details"):
                    st.session_state.show_task_modal = False
                    st.session_state.selected_task = None
                    st.session_state.last_clicked_event = None
            
            with col2:
                # Sync to Google Calendar button for tasks
                if (source == 'task' and GOOGLE_CALENDAR_AVAILABLE and 
                    st.session_state.google_calendar_service and
                    st.button("📅 Sync to Google Calendar")):
                    
                    with st.spinner("Syncing to Google Calendar..."):
                        event_id = st.session_state.google_calendar_service.sync_task_to_calendar(
                            event_info
                        )
                        if event_id:
                            st.success(f"✅ Task synced to Google Calendar!")
                        else:
                            st.error("❌ Failed to sync task to Google Calendar")

else:
    if st.session_state.calendar_type == "unified":
        st.warning("No events found. Please process emails or connect to Google Calendar.")
    elif st.session_state.calendar_type == "tasks":
        st.warning("No tasks found. Please upload and process emails from the main page first.")
    elif st.session_state.calendar_type == "google":
        st.warning("No Google Calendar events found. Please connect to Google Calendar and import events.")
    
    st.info("💡 Go to the main page to upload your Gmail Takeout ZIP file and process emails.")
