import streamlit as st
from streamlit_calendar import calendar
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


def load_tasks_from_session():
    """Load and validate tasks from session state"""


# Try to load extracted tasks from session state
if (hasattr(st.session_state, 'processing_complete') and
        st.session_state.processing_complete):
    # Extract valid tasks from the outputs
    outputs = st.session_state.get("extracted_tasks", [])
    tasks = [
        res["validated_json"] for res in outputs
        if "validated_json" in res and res.get("valid", False)
    ]
    
    if not tasks:
        st.warning("No valid tasks found in processed data.")
    else:
        st.info(f"📊 Showing {len(tasks)} tasks from processed emails")
else:
    tasks = None

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
                
            # Navigate the nested structure: validated_json > Topic > tasks
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
                start_date = task.get("start_date")
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
                event_date = due_date or start_date
                
                if event_date:  # Only add events with valid dates
                    # Create unique event ID
                    event_id = f"task_{task_index}_{len(events)}"
                    
                    # Determine event type and styling
                    is_start_event = bool(start_date)
                    # Blue for start, Orange for due
                    event_color = "#1976d2" if is_start_event else "#f57c00"
                
                    event = {
                        "id": event_id,
                        "title": task_name,
                        "start": event_date,
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
                        "start_date": start_date,
                        "due_date": due_date,
                        "owner_name": owner_name,
                        "owner_role": owner_role,
                        "owner_department": owner_dept,
                        "owner_organization": owner_org,
                        "email_index": email_index,
                        "collaborators": collaborators,
                        "event_type": "Start" if start_date else "Due"
                    }
                    
        except Exception as e:
            st.error(f"Error processing task: {e}")
            continue

    calendar_options = {
        "initialView": "dayGridMonth",
        "editable": False,
        "selectable": True,
        "selectMirror": True,
        "dayMaxEvents": 3,
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
                
                # Dates with bullet points
                if task_info['start_date']:
                    st.markdown(f"• **Start Date:** {task_info['start_date']}")
                if task_info['due_date']:
                    st.markdown(f"• **Due Date:** {task_info['due_date']}")
                
                # Summary
                if task_info['summary']:
                    st.markdown(f"• **Summary:** {task_info['summary']}")
                
                # Email Index
                if task_info.get('email_index'):
                    st.markdown(f"• **Email Index:** {task_info['email_index']}")
                
                # Responsible To (Owner) - with complete role details
                owner_name = task_info['owner_name']
                owner_role = task_info.get('owner_role', 'Unknown')
                owner_dept = task_info.get('owner_department', 'Unknown') 
                owner_org = task_info.get('owner_organization', 'Unknown')
                
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
                
                # Collaborated By (Collaborators) - with complete details for each
                if task_info.get('collaborators'):
                    collab_list = []
                    for collab in task_info['collaborators']:
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
