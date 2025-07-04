import streamlit as st
import pandas as pd
from streamlit_calendar import calendar
import datetime
import json

st.set_page_config(page_title="🗓 Calendar View", layout="wide")

# Custom CSS for Google Calendar-like styling
st.markdown("""
<style>
    .task-event {
        border-radius: 4px !important;
        font-size: 12px !important;
        padding: 2px 4px !important;
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
</style>
""", unsafe_allow_html=True)

st.title("🗓 Task Calendar View")

# Initialize session state for selected task
if "selected_task" not in st.session_state:
    st.session_state.selected_task = None

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
    st.subheader("� Task Calendar")
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
            topic_name = topic.get("name", "Unknown Topic") if isinstance(topic, dict) else "Unknown Topic"
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
                
                # Use due_date or start_date for the calendar event
                event_date = due_date or start_date
                
                if event_date:  # Only add events with valid dates
                    # Create unique event ID
                    event_id = f"task_{task_index}_{len(events)}"
                    
                    # Determine event type and styling
                    is_start_event = bool(start_date)
                    event_color = "#1976d2" if is_start_event else "#f57c00"  # Blue for start, Orange for due
                    
                    event = {
                        "id": event_id,
                        "title": task_name,
                        "start": event_date,
                        "end": event_date,
                        "allDay": True,
                        "backgroundColor": event_color,
                        "borderColor": event_color,
                        "textColor": "#ffffff",
                        "classNames": ["task-event"],
                        "extendedProps": {
                            "task_type": "Start" if is_start_event else "Due",
                            "owner": owner_name,
                            "topic": topic_name,
                            "summary": summary[:50] + "..." if len(summary) > 50 else summary
                        }
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
                        "owner_dept": owner_dept,
                        "owner_org": owner_org,
                        "email_index": email_index,
                        "event_type": "Start" if start_date else "Due"
                    }
                    
        except Exception as e:
            st.error(f"Error processing task: {e}")
            continue

    calendar_options = {
        "initialView": "dayGridMonth",
        "editable": False,
        "selectable": True,
        "displayEventTime": True,
        "eventDisplay": "block",
        "dayMaxEvents": 3,
        "moreLinkClick": "popover",
        "headerToolbar": {
            "start": "prev,next today",
            "center": "title",
            "end": "dayGridMonth,timeGridWeek,timeGridDay"
        },
        "height": "auto",
        "aspectRatio": 1.35,
        "eventClick": {
            "enabled": True
        },
        "eventMouseEnter": {
            "enabled": True
        }
    }

    # Show summary stats
    st.info(f"� Displaying {len(events)} tasks from {len(tasks)} processed emails")
    
    # Create layout: full-width calendar with side panel for details
    if events:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Render the calendar
            calendar_result = calendar(
                events=events, 
                options=calendar_options, 
                key="calendar"
            )
            
            # Handle calendar interactions
            if calendar_result.get("eventClick"):
                event_id = calendar_result["eventClick"]["event"]["id"]
                st.session_state.selected_task = event_id
                st.rerun()
        
        with col2:
            st.markdown("### 📋 Task Details")
            
            if (st.session_state.selected_task and 
                st.session_state.selected_task in task_details):
                task_info = task_details[st.session_state.selected_task]
                
                # Task header
                st.markdown(f"**{task_info['task_name']}**")
                st.caption(f"{task_info['event_type']} Date • {task_info['topic']}")
                
                # Key information in compact format
                if task_info['summary']:
                    st.markdown(f"📝 {task_info['summary']}")
                
                # Dates
                col_start, col_due = st.columns(2)
                with col_start:
                    if task_info['start_date']:
                        st.metric("Start", task_info['start_date'])
                with col_due:
                    if task_info['due_date']:
                        st.metric("Due", task_info['due_date'])
                
                # Owner info
                st.markdown(f"👤 **{task_info['owner_name']}**")
                if task_info['owner_role']:
                    st.caption(task_info['owner_role'])
                
                # Action button
                if st.button("🔍 More Context", 
                           key=f"ctx_{st.session_state.selected_task}"):
                    from utils.graphrag import (GraphRAG, 
                                              format_graphrag_response)
                    
                    try:
                        graphrag = GraphRAG()
                        query = (f"Tell me about task "
                               f"'{task_info['task_name']}'")
                        result = graphrag.query_with_semantic_reasoning(query)
                        formatted = format_graphrag_response(result)
                        
                        with st.expander("🧠 AI Context", expanded=True):
                            st.markdown(formatted)
                            
                    except Exception as e:
                        st.error(f"Context error: {e}")
            else:
                st.info("👆 Click a task to see details")
                
                # Quick task list
                if events:
                    st.markdown("**All Tasks:**")
                    for event in events[:5]:  # Show first 5
                        event_id = event.get("id")
                        if event_id in task_details:
                            task_info = task_details[event_id]
                            if st.button(
                                f"📌 {task_info['task_name'][:25]}...", 
                                key=f"sel_{event_id}"
                            ):
                                st.session_state.selected_task = event_id
                                st.rerun()
                    
                    if len(events) > 5:
                        st.caption(f"... and {len(events) - 5} more tasks")
    else:
        st.warning("No tasks with valid dates found.")

    # Optional: Show technical details for debugging
    with st.expander("� Technical Details", expanded=False):
        st.write(f"**Total events:** {len(events)}")
        st.write(f"**Events with dates:** {len([e for e in events if e.get('start')])}")
        
        if events:
            st.write("**Sample events:**")
            for i, event in enumerate(events[:3]):
                st.write(f"• {event['title']} - {event['start']}")