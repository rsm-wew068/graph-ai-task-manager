
import streamlit as st
from streamlit_calendar import calendar
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.notion_utils import query_tasks, is_configured

st.set_page_config(page_title="🗓 Notion Task Calendar", layout="wide")

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

# Initialize session state
if "selected_task" not in st.session_state:
    st.session_state.selected_task = None
if "show_task_modal" not in st.session_state:
    st.session_state.show_task_modal = False
if "last_clicked_event" not in st.session_state:
    st.session_state.last_clicked_event = None
if "calendar_type" not in st.session_state:
    st.session_state.calendar_type = "tasks"


# Query tasks from Notion
events = []
task_details = {}

if not is_configured():
    st.warning("Notion API key or database ID not set. Please set NOTION_API_KEY and NOTION_DATABASE_ID in your environment.")
else:
    notion_tasks = query_tasks(page_size=100)
    st.markdown("### 📋 Notion Tasks")
    for idx, page in enumerate(notion_tasks):
        props = page.get("properties", {})
        
        # Safe property extraction with null checks
        name_prop = props.get("Name", {})
        name = (name_prop.get("title", [{}])[0].get("text", {}).get("content", "Unnamed Task") 
                if name_prop.get("title") else "Unnamed Task")
        
        due_date_prop = props.get("Due Date", {})
        due_date = due_date_prop.get("date", {}).get("start") if due_date_prop.get("date") else None
        
        received_date_prop = props.get("Received Date", {})
        received_date = received_date_prop.get("date", {}).get("start") if received_date_prop.get("date") else None
        
        summary_prop = props.get("Task Description", {})
        summary = (summary_prop.get("rich_text", [{}])[0].get("text", {}).get("content", "") 
                  if summary_prop.get("rich_text") else "")
        
        status_prop = props.get("Status", {})
        status = status_prop.get("status", {}).get("name", "") if status_prop.get("status") else ""
        
        topic_prop = props.get("Topic", {})
        topic = (topic_prop.get("select", {}).get("name", "") 
                if topic_prop.get("select") else "")
        
        priority_prop = props.get("Priority Level", {})
        priority = (priority_prop.get("select", {}).get("name", "") 
                   if priority_prop.get("select") else "")
        
        sender_prop = props.get("Sender", {})
        sender = sender_prop.get("email", "") if sender_prop.get("email") else ""
        
        assigned_to_prop = props.get("Assigned To", {})
        assigned_to = assigned_to_prop.get("email", "") if assigned_to_prop.get("email") else ""
        
        email_source_prop = props.get("Email Source", {})
        email_source = (email_source_prop.get("url", "") 
                       if email_source_prop.get("url") else "")
        
        event_id = f"notion_{page['id']}"
        event_date = due_date or received_date
        if event_date:
            event = {
                "id": event_id,
                "title": f"{name} 📋",
                "start": event_date,
                "allDay": True,
                "backgroundColor": "#1976d2",
                "borderColor": "#1976d2",
                "textColor": "#ffffff"
            }
            events.append(event)
            task_details[event_id] = {
                "task_name": name,
                "summary": summary,
                "start_date": received_date,
                "due_date": due_date,
                "status": status,
                "topic": topic,
                "priority": priority,
                "sender": sender,
                "assigned_to": assigned_to,
                "email_source": email_source,
                "source": "notion"
            }



# Display calendar

if events:
    st.metric("Total Notion Tasks", len(events))
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
    if calendar_result.get("eventClick"):
        event_info = calendar_result["eventClick"]["event"]
        event_id = event_info.get("id")
        if event_id != st.session_state.last_clicked_event:
            st.session_state.last_clicked_event = event_id
            if event_id and event_id in task_details:
                st.session_state.selected_task = event_id
                st.session_state.show_task_modal = True
    if (st.session_state.show_task_modal and
        st.session_state.selected_task and
        st.session_state.selected_task in task_details):
        event_info = task_details[st.session_state.selected_task]
        with st.expander("📝 Event Details", expanded=True):
            st.markdown(f"**Event:** {event_info['task_name']}")
            st.markdown(f"**Status:** {event_info.get('status', 'N/A')}")
            st.markdown(f"**Topic:** {event_info.get('topic', 'N/A')}")
            if event_info.get('start_date'):
                st.markdown(f"• **Received Date:** {event_info['start_date']}")
            if event_info.get('due_date'):
                st.markdown(f"• **Due Date:** {event_info['due_date']}")
            if event_info.get('summary'):
                st.markdown(f"• **Summary:** {event_info['summary']}")
            if event_info.get('priority'):
                st.markdown(f"• **Priority:** {event_info['priority']}")
            if event_info.get('sender'):
                st.markdown(f"• **Sender:** {event_info['sender']}")
            if event_info.get('assigned_to'):
                st.markdown(f"• **Assigned To:** {event_info['assigned_to']}")
            if event_info.get('email_source'):
                st.markdown(f"• **Email Source:** {event_info['email_source']}")
            if st.button("✖ Close", key="close_event_details"):
                st.session_state.show_task_modal = False
                st.session_state.selected_task = None
                st.session_state.last_clicked_event = None
else:
    st.warning("No Notion tasks found. Please add tasks to your Notion database.")
