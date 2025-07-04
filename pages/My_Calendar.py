import streamlit as st
import pandas as pd
from streamlit_calendar import calendar
import datetime

st.set_page_config(page_title="🗓 Calendar View", layout="wide")
st.title("🗓 Task Calendar View")

# Try to load extracted tasks from session state
tasks = st.session_state.get("validated_tasks")

if tasks is None:
    st.warning("No tasks found. Please upload emails from the main page first.")
else:
    st.subheader("📆 Task Calendar")

    # Convert tasks to event format
    events = []
    for task in tasks:
        try:
            event = {
                "title": task.get("deliverable", "Unnamed Task"),
                "start": task.get("due_date") or task.get("start_date"),
                "end": task.get("due_date") or task.get("start_date"),
                "description": task["owner"]["name"] if isinstance(task["owner"], dict) else str(task["owner"]),
            }
            events.append(event)
        except Exception:
            continue

    calendar_options = {
        "initialView": "dayGridMonth",
        "editable": False,
        "selectable": False,
        "headerToolbar": {
            "start": "title",
            "center": "",
            "end": "today prev,next"
        }
    }

    calendar(events=events, options=calendar_options)