#!/usr/bin/env python3
"""
Automated Task Manager - Main Streamlit Application
Production-ready with Neo4j, persistent storage, and full data processing
"""

import streamlit as st
import os
import pandas as pd
import json
from pathlib import Path
import tempfile
import zipfile

# Page configuration
st.set_page_config(
    page_title="Automated Task Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_environment():
    """Check environment variable status for production deployment"""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        st.error("""
        🚨 **Missing OPENAI_API_KEY Environment Variable**
        
        The task extraction requires an OpenAI API key to function.
        
        **For production deployment:**
        - Set the environment variable in your deployment environment
        - Or create a `.env` file with `OPENAI_API_KEY=your_key_here`
        """)
        return False
    return True

def flatten_extractions(json_list):
    """
    Flatten extracted JSON data into a user-friendly table format.
    
    Args:
        json_list: List of validated JSON objects from extraction
        
    Returns:
        pandas.DataFrame: Flattened task data
    """
    rows = []
    for item in json_list:
        # Handle both dict and string cases
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except (json.JSONDecodeError, TypeError):
                # Skip invalid items
                continue
        
        # Skip non-dict items (None, etc.)
        if not isinstance(item, dict):
            continue
        
        # Handle validated_json wrapper from HITL validation
        if "validated_json" in item:
            item = item["validated_json"]
            
        topic = item.get("Topic", {})
        topic_name = (topic.get("name", "Unknown Topic")
                      if isinstance(topic, dict) else "Unknown Topic")
        
        tasks = topic.get("tasks", []) if isinstance(topic, dict) else []
        for task_obj in tasks:
            if not isinstance(task_obj, dict):
                continue
                
            task = task_obj.get("task", {})
            if not isinstance(task, dict):
                continue
                
            # Extract comprehensive owner information
            owner = task.get("owner", {})
            if isinstance(owner, dict):
                owner_name = owner.get("name", "Unknown")
                owner_role = owner.get("role", "")
                owner_dept = owner.get("department", "")
                owner_org = owner.get("organization", "")
            else:
                owner_name = str(owner) if owner else "Unknown"
                owner_role = owner_dept = owner_org = ""
            
            # Extract all task details for comprehensive display
            rows.append({
                "Topic": topic_name,
                "Task Name": task.get("name", "Unnamed Task"),
                "Summary": task.get("summary", ""),
                "Start Date": task.get("start_date", ""),
                "Due Date": task.get("due_date", ""),
                "Owner Name": owner_name,
                "Owner Role": owner_role,
                "Owner Department": owner_dept,
                "Owner Organization": owner_org,
                "Email Index": task_obj.get("email_index", "")
            })
    return pd.DataFrame(rows)

def main():
    """Main Streamlit application."""
    import pandas as pd
    
    # Configure Streamlit for better persistence
    st.set_page_config(
        page_title="Automated Task Manager",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check environment
    env_status = check_environment()
    if not env_status:
        st.warning("⚠️ Application may not work correctly without proper API keys")
        st.info("You can still upload and parse emails, but extraction will fail")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    .upload-section {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #4A90E2;
        text-align: center;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Automated Task Manager</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Email Task Management with Neo4j & LangGraph</p>', unsafe_allow_html=True)
    
    # Initialize session state for persistence
    if "parsed_emails" not in st.session_state:
        st.session_state.parsed_emails = []
    if "extracted_tasks" not in st.session_state:
        st.session_state.extracted_tasks = []
    if "parsing_complete" not in st.session_state:
        st.session_state.parsing_complete = False
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "extracted_graphs" not in st.session_state:
        st.session_state.extracted_graphs = []
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if "filter_settings" not in st.session_state:
        st.session_state.filter_settings = None
    
    # Sidebar for system status
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        # Gmail integration status
        try:
            from utils.gmail_service import GmailService
            gmail_service = GmailService()
            if gmail_service.authenticate():
                st.success("✅ Gmail API Connected")
            else:
                st.info("📧 Gmail API Available (not connected)")
        except:
            st.info("📧 Gmail API Not Configured")
        
        # Check if Neo4j is connected
        try:
            from utils.api import driver
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()["count"]
                st.metric("Total Nodes", node_count)
                st.success("✅ Neo4j Connected")
        except Exception as e:
            st.metric("Total Nodes", "Not Connected")
            st.error("⚠️ Neo4j connection failed")
        
        # Check FastAPI backend
        try:
            import requests
            response = requests.get("http://localhost:8000/", timeout=2)
            if response.status_code == 200:
                st.success("✅ FastAPI Backend Connected")
            else:
                st.warning("⚠️ FastAPI Backend Issues")
        except:
            st.error("❌ FastAPI Backend Not Available")
        
        st.markdown("### 🔧 System Info")
        st.info("""
        - **Frontend**: Streamlit ✅
        - **Backend**: FastAPI (Chat History)
        - **Database**: Neo4j Graph Database
        - **AI Engine**: LangChain + LangGraph
        - **Storage**: Persistent Neo4j
        - **Deployment**: Production Ready
        """)
        
        # Show processing status
        if hasattr(st.session_state, 'processing_complete') and st.session_state.processing_complete:
            outputs = st.session_state.get("extracted_tasks", [])
            valid_tasks_count = len([
                res for res in outputs
                if "validated_json" in res and res.get("valid", False)
            ])
            st.success(f"✅ {valid_tasks_count} tasks processed")
        elif hasattr(st.session_state, 'parsing_complete') and st.session_state.parsing_complete:
            st.warning("📁 Emails parsed, ready for processing")
        else:
            st.info("📝 No data loaded yet")
        
        # Clear data button
        if st.button("🗑️ Clear Cached Data"):
            for key in ["parsed_emails", "extracted_tasks", "parsing_complete", 
                        "processing_complete", "uploaded_file_name", "parse_limit",
                        "process_limit", "extracted_graphs"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Cached data cleared!")
            st.rerun()

        # Clear database buttons
        st.subheader("🗑️ Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗄️ Clear Neo4j Database"):
                try:
                    from neo4j import GraphDatabase
                    import os
                    from dotenv import load_dotenv
                    load_dotenv()
                    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
                    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
                    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
                    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                    with driver.session() as session:
                        session.run("MATCH (n) DETACH DELETE n")
                    driver.close()
                    st.success("✅ Neo4j database cleared!")
                except Exception as e:
                    st.error(f"❌ Failed to clear Neo4j database: {e}")
        
        with col2:
            if st.button("📝 Clear Notion Database"):
                try:
                    from utils.notion_utils import clear_notion_database
                    success = clear_notion_database()
                    if success:
                        st.success("✅ Notion database cleared!")
                    else:
                        st.error("❌ Failed to clear Notion database")
                except Exception as e:
                    st.error(f"❌ Failed to clear Notion database: {e}")
    
    # Add in-app help/documentation button at the top of the main page
    with st.expander("❓ Help & Instructions", expanded=False):
            st.markdown("""
        ### How to Use the Automated Task Manager
        
        **1. Connect Your Email**
        - Go to the sidebar and click **Connect to Gmail**.
        - Follow the authentication steps to allow access to your inbox.
        
        **2. Load and Filter Emails**
        - Use the **Gmail Integration** tab to load recent emails.
        - Adjust filters (date range, unread, keywords) to focus on relevant emails.
        
        **3. Extract Tasks**
        - Click **Extract Tasks (Enhanced)** to use AI to find tasks in your emails.
        - You can also extract tasks from individual emails using the button in each email card.
        
        **4. Manage and Validate Tasks**
        - Go to the **Task Management** tab to review, validate, and download extracted tasks.
        - Use the human validation interface for any tasks that need review.
        
        **5. Analyze Results**
        - The **Results Analysis** tab shows statistics and detailed analysis of extraction results.
        
        **Smart Filters**
        - Smart filters use date range, keywords, content length, and email type to prioritize relevant emails.
        - You can customize these filters in the sidebar or upcoming user settings.
        
        **Need more help?**
        - Reach out to the project maintainer or check the documentation for advanced usage.
        """)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📧 Gmail Integration", "📋 Task Management", "🔍 Results Analysis"])
    
    with tab1:
        # --- Begin Gmail Integration code (from pages/Gmail_Integration.py) ---
        import pandas as pd
        from datetime import datetime, timedelta
        import time
        try:
            from utils.gmail_service import GmailService
            from utils.gmail_realtime_processor import create_realtime_processor
            from utils.gmail_smart_filters import create_user_settings_filter, create_meeting_focused_filter, create_project_focused_filter
            from utils.gmail_enhanced_extraction import create_enhanced_extractor
            GMAIL_AVAILABLE = True
        except ImportError as e:
            st.error(f"Import error: {e}")
            GMAIL_AVAILABLE = False

        # --- User Settings for Smart Filter ---
        st.markdown("### ⚙️ Smart Filter Settings")
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now()),
                help="Only include emails within this date range."
            )
            min_length = st.number_input(
                "Minimum Content Length",
                min_value=0,
                max_value=10000,
                value=50,
                help="Exclude emails with body shorter than this many characters."
            )
            max_length = st.number_input(
                "Maximum Content Length",
                min_value=0,
                max_value=100000,
                value=5000,
                help="Exclude emails with body longer than this many characters."
            )
        with col2:
            keywords = st.text_input(
                "Keywords (comma-separated)",
                value="task,todo,deadline,meeting,project,action",
                help="Only include emails containing these keywords in subject or body."
            )
            email_types = st.multiselect(
                "Email Types",
                ["Primary", "Promotions", "Social", "Updates"],
                default=["Primary"],
                help="Include emails from these Gmail categories."
            )
        # Store settings in session state
        st.session_state.smart_filter_settings = {
            "date_range": date_range,
            "min_length": min_length,
            "max_length": max_length,
            "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
            "email_types": email_types
        }

        # Create dynamic smart filter from user settings
        user_smart_filter = create_user_settings_filter(st.session_state.smart_filter_settings)
        st.session_state.smart_filter = user_smart_filter

        # --- Analytics/Usage Stats ---
        st.markdown("### 📊 Analytics & Usage Stats")
        emails_loaded = len(st.session_state.get('emails_data', []))
        emails_filtered = emails_loaded  # Already filtered by smart filter
        tasks_extracted = len(st.session_state.get('processed_emails', []))
        success_rate = (tasks_extracted / emails_filtered * 100) if emails_filtered else 0
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Emails Loaded", emails_loaded)
        with col2:
            st.metric("After Filtering", emails_filtered)
        with col3:
            st.metric("Tasks Extracted", tasks_extracted)
        with col4:
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # --- Send Test Email (SMTP) ---
        st.markdown("### 📤 Send Test Email (SMTP)")
        with st.form("send_test_email_form"):
            smtp_to = st.text_input("Recipient Email", "test@example.com")
            smtp_subject = st.text_input("Subject", "Test Email from Automated Task Manager")
            smtp_body = st.text_area("Body", "This is a test email sent from the Automated Task Manager app.")
            submitted = st.form_submit_button("Send Email")
        if submitted:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            import os
            import streamlit as st
            # Try to use Gmail credentials from token if available
            gmail_user = None
            gmail_pass = None
            # Try to get from environment or session (for demo, user must provide app password if 2FA enabled)
            gmail_user = st.session_state.gmail_service.credentials._id_token.get('email') if hasattr(st.session_state.gmail_service, 'credentials') and st.session_state.gmail_service.credentials and hasattr(st.session_state.gmail_service.credentials, '_id_token') and st.session_state.gmail_service.credentials._id_token else os.getenv('GMAIL_USER')
            gmail_pass = os.getenv('GMAIL_APP_PASSWORD')
            if not gmail_user or not gmail_pass:
                st.error("Gmail address and app password required. Set GMAIL_USER and GMAIL_APP_PASSWORD as environment variables.")
            else:
                try:
                    msg = MIMEMultipart()
                    msg['From'] = gmail_user
                    msg['To'] = smtp_to
                    msg['Subject'] = smtp_subject
                    msg.attach(MIMEText(smtp_body, 'plain'))
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(gmail_user, gmail_pass)
                    server.sendmail(gmail_user, smtp_to, msg.as_string())
                    server.quit()
                    st.success(f"Email sent to {smtp_to}!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
        
        # Custom CSS for better styling (from Gmail_Integration)
        st.markdown("""
        <style>
        .gmail-header { background: linear-gradient(135deg, #4285f4, #34a853, #fbbc05, #ea4335); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; }
        .email-card { background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .email-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); transform: translateY(-2px); transition: all 0.3s ease; }
        .status-badge { padding: 0.25rem 0.5rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold; }
        .status-unread { background: #e3f2fd; color: #1976d2; }
        .status-read { background: #f5f5f5; color: #666; }
        .task-extracted { background: #e8f5e8; color: #2e7d32; border-left: 4px solid #4caf50; }
        .metric-card { background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-number { font-size: 2rem; font-weight: bold; color: #4285f4; }
        .metric-label { color: #666; font-size: 0.9rem; margin-top: 0.5rem; }
        .feature-card { background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .feature-card h3 { color: #4285f4; margin-bottom: 1rem; }
        .filter-score { background: #fff3e0; color: #f57c00; padding: 0.25rem 0.5rem; border-radius: 10px; font-size: 0.8rem; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="gmail-header">
            <h1>📧 Gmail Integration</h1>
            <p>Connect to your Gmail account and extract tasks from emails with AI-powered intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize session state (copied from Gmail_Integration)
        if "gmail_service" not in st.session_state:
            st.session_state.gmail_service = None
        if "gmail_authenticated" not in st.session_state:
            st.session_state.gmail_authenticated = False
        if "emails_data" not in st.session_state:
            st.session_state.emails_data = []
        if "processed_emails" not in st.session_state:
            st.session_state.processed_emails = []
        if "realtime_processor" not in st.session_state:
            st.session_state.realtime_processor = None
        if "enhanced_extractor" not in st.session_state:
            st.session_state.enhanced_extractor = None
        if "smart_filter" not in st.session_state:
            st.session_state.smart_filter = None

        # Sidebar for Gmail controls (copied from Gmail_Integration)
        with st.sidebar:
            st.markdown("### 🔐 Gmail Authentication")
            if not GMAIL_AVAILABLE:
                st.error("❌ Gmail service not available")
                st.info("Please ensure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are set in your .env file")
            else:
                if st.session_state.gmail_service is None:
                    st.session_state.gmail_service = GmailService()
                if st.button("🔐 Connect to Gmail", use_container_width=True):
                    with st.spinner("Connecting to Gmail..."):
                        if st.session_state.gmail_service.authenticate():
                            st.session_state.gmail_authenticated = True
                            st.success("✅ Connected to Gmail!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to connect to Gmail")
                if st.session_state.gmail_authenticated:
                    st.success("✅ Gmail Connected")
                    if st.session_state.enhanced_extractor is None:
                        st.session_state.enhanced_extractor = create_enhanced_extractor(st.session_state.gmail_service, st.session_state.smart_filter)
                    if st.session_state.realtime_processor is None:
                        st.session_state.realtime_processor = create_realtime_processor(st.session_state.gmail_service)
                    st.markdown("### 🎯 Smart Filtering")
                    filter_type = st.selectbox(
                        "Choose Filter Type",
                        ["Task Focused", "Meeting Focused", "Project Focused", "Custom"]
                    )
                    if filter_type == "Task Focused":
                        st.session_state.smart_filter = create_user_settings_filter(st.session_state.smart_filter_settings)
                    elif filter_type == "Meeting Focused":
                        st.session_state.smart_filter = create_meeting_focused_filter()
                    elif filter_type == "Project Focused":
                        st.session_state.smart_filter = create_project_focused_filter()
                    st.markdown("### ⚡ Real-time Processing")
                    if st.session_state.realtime_processor.is_running:
                        if st.button("⏹️ Stop Real-time Processing", use_container_width=True):
                            st.session_state.realtime_processor.stop_monitoring()
                            st.success("⏹️ Real-time processing stopped")
                            st.rerun()
                    else:
                        if st.button("▶️ Start Real-time Processing", use_container_width=True):
                            def task_callback(extracted_tasks):
                                st.session_state.processed_emails.extend(extracted_tasks)
                            st.session_state.realtime_processor.start_monitoring(
                                callback=task_callback,
                                interval=300
                            )
                            st.success("▶️ Real-time processing started")
                            st.rerun()
                    if st.button("🔄 Refresh Emails", use_container_width=True):
                        with st.spinner("Fetching emails..."):
                            recent_emails = st.session_state.gmail_service.get_recent_emails(hours=168)
                            # Apply user smart filter
                            emails = st.session_state.smart_filter.filter_emails(recent_emails)
                            st.session_state.emails_data = emails
                            st.success(f"✅ Fetched {len(emails)} emails (filtered)")
                            st.rerun()
                if st.session_state.emails_data:
                    st.markdown("### 📊 Email Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Emails", len(st.session_state.emails_data))
                    with col2:
                        unread_count = len([e for e in st.session_state.emails_data if 'UNREAD' in e.get('labels', [])])
                        st.metric("Unread", unread_count)
                    if st.session_state.enhanced_extractor:
                        summary = st.session_state.enhanced_extractor.get_extraction_summary()
                        st.markdown("### 📈 Extraction Summary")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processed", summary['total_processed'])
                        with col2:
                            st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                # Main content area (copied from Gmail_Integration)
                if not GMAIL_AVAILABLE:
                    st.error("❌ Gmail integration is not available")
                    st.info("""
                    **To enable Gmail integration:**
                    1. Set up Google Cloud Project with Gmail API
                    2. Configure OAuth 2.0 credentials
                    3. Set environment variables: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET
                    4. Download credentials.json from Google Cloud Console
                    """)
                else:
                    if not st.session_state.gmail_authenticated:
                        st.info("🔐 Please connect to Gmail using the sidebar to start using the integration.")
                        st.markdown("### 🚀 Available Features")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>🤖 AI-Powered Task Extraction</h3>
                                <p>Extract tasks, deadlines, and action items from emails using advanced AI</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>⚡ Real-time Processing</h3>
                                <p>Automatically process new emails as they arrive in your inbox</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>🎯 Smart Filtering</h3>
                                <p>Intelligent filtering to focus on the most relevant emails</p>
                            </div>
                            """, unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>📊 Enhanced Analytics</h3>
                                <p>Detailed insights into email processing and task extraction</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>🏷️ Label Management</h3>
                                <p>Organize emails with custom labels and categories</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>📧 Email Actions</h3>
                                <p>Reply, forward, and manage emails directly from the interface</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ Connected to Gmail! You can now view and process your emails.")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            email_filter = st.selectbox(
                                "Filter Emails",
                                ["All Recent", "Unread Only", "Last 24 Hours", "Last 7 Days", "Search", "Smart Filtered"]
                            )
                        with col2:
                            if email_filter == "Search":
                                search_query = st.text_input("Search Query", placeholder="from:example.com subject:meeting")
                            else:
                                search_query = ""
                        with col3:
                            max_results = st.slider("Max Results", 10, 100, 50)
                        if st.button("📥 Load Emails", use_container_width=True):
                            with st.spinner("Fetching emails..."):
                                if email_filter == "Unread Only":
                                    emails = st.session_state.gmail_service.get_unread_emails(max_results)
                                elif email_filter == "Search" and search_query:
                                    emails = st.session_state.gmail_service.search_emails(search_query, max_results)
                                elif email_filter == "Smart Filtered":
                                    recent_emails = st.session_state.gmail_service.get_recent_emails(hours=168)
                                    emails = st.session_state.smart_filter.filter_emails(recent_emails)
                                    emails = emails[:max_results]
                                else:
                                    hours = 24 if email_filter == "Last 24 Hours" else (168 if email_filter == "Last 7 Days" else 24)
                                    emails = st.session_state.gmail_service.get_recent_emails(hours)
                                    emails = emails[:max_results]
                                # Always apply user smart filter if not already applied
                                if email_filter not in ["Smart Filtered", "Search"]:
                                    emails = st.session_state.smart_filter.filter_emails(emails)
                                st.session_state.emails_data = emails
                                st.success(f"✅ Loaded {len(emails)} emails (filtered)")
                                st.rerun()
                        if st.session_state.emails_data:
                            st.markdown(f"### 📧 Emails ({len(st.session_state.emails_data)} found)")
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                if st.button("🤖 Extract Tasks (Enhanced)", use_container_width=True):
                                    with st.spinner("Processing emails with enhanced extraction..."):
                                        if st.session_state.enhanced_extractor:
                                            results = st.session_state.enhanced_extractor.extract_tasks_from_emails(
                                                st.session_state.emails_data, 
                                                max_emails=50
                                            )
                                            st.session_state.processed_emails.extend(results)
                                            st.success(f"✅ Extracted tasks from {len(results)} emails")
                                            st.rerun()
                                        else:
                                            st.error("❌ Enhanced extractor not available")
                            with col2:
                                if st.button("🏷️ Manage Labels", use_container_width=True):
                                    st.info("Label management feature coming soon!")
                            with col3:
                                if st.button("📊 View Analytics", use_container_width=True):
                                    st.info("Analytics dashboard coming soon!")
                            for i, email in enumerate(st.session_state.emails_data):
                                is_unread = 'UNREAD' in email.get('labels', [])
                                status_class = "status-unread" if is_unread else "status-read"
                                filter_score = email.get('filter_score', 0)
                                with st.expander(f"📧 {email['subject']}", expanded=is_unread):
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**From:** {email['sender']}")
                                        st.markdown(f"**Date:** {email['date']}")
                                        st.markdown(f"**Snippet:** {email['snippet']}")
                                        if filter_score > 0:
                                            st.markdown(f'<span class="filter-score">Filter Score: {filter_score:.1f}</span>', unsafe_allow_html=True)
                                        if email['body']:
                                            body_preview = email['body'][:200] + "..." if len(email['body']) > 200 else email['body']
                                            st.markdown(f"**Body:** {body_preview}")
                                    with col2:
                                        st.markdown(f'<span class="status-badge {status_class}">{"📬 Unread" if is_unread else "📭 Read"}</span>', unsafe_allow_html=True)
                                        if st.button(f"📋 Extract Tasks", key=f"extract_{i}"):
                                            with st.spinner("Extracting tasks..."):
                                                if st.session_state.enhanced_extractor:
                                                    results = st.session_state.enhanced_extractor.extract_tasks_from_emails([email])
                                                    if results:
                                                        st.session_state.processed_emails.extend(results)
                                                        st.success("✅ Tasks extracted successfully!")
                                                        st.rerun()
                                                    else:
                                                        st.warning("⚠️ No tasks found in this email")
                                                else:
                                                    st.error("❌ Enhanced extractor not available")
                                        if st.button(f"📧 Reply", key=f"reply_{i}"):
                                            st.info("Email reply feature coming soon!")
                                        if st.button(f"🏷️ Label", key=f"label_{i}"):
                                            st.info("Label management feature coming soon!")
                        else:
                            st.info("📧 No emails loaded. Click 'Load Emails' to fetch your recent emails.")
                        if st.session_state.processed_emails:
                            st.markdown("---")
                            st.markdown(f"### 📋 Processed Emails ({len(st.session_state.processed_emails)})")
                            for i, processed in enumerate(st.session_state.processed_emails[-10:]):
                                with st.expander(f"📋 {processed.get('subject', 'Unknown Subject')}", expanded=False):
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**Sender:** {processed.get('sender', 'Unknown')}")
                                        st.markdown(f"**Date:** {processed.get('date', 'Unknown')}")
                                        st.markdown(f"**Status:** {processed.get('status', 'Unknown')}")
                                        if processed.get('filter_score'):
                                            st.markdown(f"**Filter Score:** {processed.get('filter_score', 0):.1f}")
                                        if processed.get('extraction_result'):
                                            st.markdown("**Extraction Result:**")
                                            st.json(processed.get('extraction_result'))
                                    with col2:
                                        if processed.get('status') == 'processed':
                                            st.success("✅ Processed")

                st.markdown("---")
                st.markdown("### 🚀 Coming Soon")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="feature-card">
                        <h3>📅 Calendar Integration</h3>
                        <p>Automatically create calendar events from meeting emails</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="feature-card">
                        <h3>🤖 AI Email Assistant</h3>
                        <p>AI-powered email drafting and response suggestions</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div class="feature-card">
                        <h3>📊 Advanced Analytics</h3>
                        <p>Detailed insights into email patterns and productivity</p>
                    </div>
                    """, unsafe_allow_html=True)
        # --- End Gmail Integration code ---
    
    with tab2:
        # Task Management Section
        st.markdown("## 📋 Task Management")
        
        if hasattr(st.session_state, 'processing_complete') and st.session_state.processing_complete:
            outputs = st.session_state.get("extracted_tasks", [])
            
            # Separate valid and invalid results
            valid_tasks = []
            invalid_results = []
            paused_results = []
            
            for i, res in enumerate(outputs):
                # Check for paused/awaiting review status (including errors that need review)
                if (res.get("status") in ["paused", "awaiting_user_review"] or 
                    res.get("needs_user_review", False)):
                    paused_results.append((i, res))
                elif "validated_json" in res and res.get("valid", False):
                    valid_tasks.append(res["validated_json"])
                else:
                    # Store invalid results for display
                    invalid_results.append({
                        "email_index": i + 1,
                        "raw_json": res.get("extracted_json", "No JSON extracted"),
                        "email_id": res.get("email_index", "Unknown"),
                        "status": res.get("status", "unknown")
                    })
            
            # Handle paused results that need human validation
            if paused_results:
                st.subheader("⏸️ Human Validation Required")
                
                for idx, result in paused_results:
                    st.markdown(f"**Email {idx + 1}** - Validation needed:")
                    
                    # Show email context for better validation
                    if result.get("email_subject"):
                        st.info(f"📧 **Subject:** {result['email_subject']}")
                    if result.get("email_content"):
                        with st.expander("📄 View Email Content", expanded=False):
                            st.text(result["email_content"])
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**Current JSON Template:**")
                        original_json = result.get("extracted_json", {})
                        st.json(original_json)
                    
                    with col2:
                        st.markdown("**Edit JSON (extract task details):**")
                        json_key = f"json_edit_{idx}"
                        
                        # Use correctable_json if available
                        correctable_template = result.get("correctable_json", 
                                                         result.get("extracted_json", "{}"))
                        
                        if json_key not in st.session_state:
                            if isinstance(correctable_template, str):
                                st.session_state[json_key] = correctable_template
                            else:
                                st.session_state[json_key] = json.dumps(
                                    correctable_template, indent=2
                                )
                        
                        corrected_json = st.text_area(
                            "Corrected JSON",
                            value=st.session_state[json_key],
                            height=300,
                            key=f"correction_{idx}",
                            help="Edit the JSON to extract the actual task from the email"
                        )
                        
                        if st.button("✅ Validate & Continue", key=f"validate_{idx}"):
                            try:
                                # Parse the corrected JSON
                                parsed_correction = json.loads(corrected_json)
                                
                                # Update the result with validated JSON
                                st.session_state.extracted_tasks[idx].update({
                                    "validated_json": parsed_correction,
                                    "valid": True,
                                    "status": "validated",
                                    "user_corrected_json": corrected_json
                                })
                                
                                # Move to valid tasks immediately
                                if "valid_extraction_results" not in st.session_state:
                                    st.session_state.valid_extraction_results = []
                                
                                st.session_state.valid_extraction_results.append(
                                    st.session_state.extracted_tasks[idx]
                                )
                                
                                st.success("✅ Task validated and moved to 'Extracted Valid Tasks'")
                                st.info("🔄 Refreshing page to show updated results...")
                                st.rerun()
                                
                            except json.JSONDecodeError as e:
                                st.error(f"❌ Invalid JSON format: {str(e)}")
                                st.info("💡 Tip: Check for missing commas, quotes, or brackets")
                            except Exception as e:
                                st.error(f"❌ Error during validation: {str(e)}")
            
            # Display valid tasks with flattened format
            # Combine original valid tasks with newly validated ones
            all_valid_tasks = valid_tasks.copy()
            if hasattr(st.session_state, 'valid_extraction_results'):
                for validated_result in st.session_state.valid_extraction_results:
                    if "validated_json" in validated_result:
                        all_valid_tasks.append(validated_result["validated_json"])
            
            if all_valid_tasks:
                st.subheader("✅ Extracted Valid Tasks")
                
                # Use the flattening function
                flattened_df = flatten_extractions(all_valid_tasks)
                st.dataframe(flattened_df, use_container_width=True)
                
                # Show summary stats
                original_count = len(valid_tasks)
                validated_count = len(all_valid_tasks) - original_count
                
                if validated_count > 0:
                    st.info(
                        f"📊 Total: {len(flattened_df)} tasks "
                        f"({original_count} auto-extracted + {validated_count} human-validated) "
                        f"from {len(outputs)} processed emails"
                    )
                else:
                    st.info(
                        f"📊 Successfully extracted {len(flattened_df)} tasks "
                        f"from {len(outputs)} processed emails"
                    )
                
                # Download option
                csv = flattened_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Tasks as CSV",
                    data=csv,
                    file_name="extracted_tasks.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No valid tasks found in the processed data")
            
            # Display invalid results (if any remain)
            if invalid_results:
                with st.expander("❌ Invalid/Failed Extractions", expanded=False):
                    st.write("These emails failed to extract valid tasks:")
                    invalid_df = pd.DataFrame(invalid_results)
                    st.dataframe(invalid_df, use_container_width=True)
        else:
            st.info("📝 No tasks available. Please upload and process emails first.")
            
            # Quick start guide
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            1. **Upload Email Data**: Go to the "Upload & Process" tab
            2. **Parse Emails**: Click "Parse Emails" to extract email data
            3. **Extract Tasks**: Click "Extract Tasks with AI" to process with LLM
            4. **View Results**: Return here to see your extracted tasks
            """)
    
    with tab3:
        # Results Analysis Section
        st.markdown("## 🔍 Results Analysis")
        
        if hasattr(st.session_state, 'processing_complete') and st.session_state.processing_complete:
            outputs = st.session_state.get("extracted_tasks", [])
            
            # Statistics
            valid_count = len([r for r in outputs if r.get("valid", False)])
            error_count = len([r for r in outputs if r.get("status") == "error"])
            paused_count = len([r for r in outputs if r.get("needs_user_review", False)])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Processed", len(outputs))
            with col2:
                st.metric("Valid Tasks", valid_count)
            with col3:
                st.metric("Errors", error_count)
            with col4:
                st.metric("Needs Review", paused_count)
            
            # Success rate
            if len(outputs) > 0:
                success_rate = (valid_count / len(outputs)) * 100
                st.info(f"📊 Success Rate: {success_rate:.1f}% ({valid_count}/{len(outputs)} emails)")
            
            # Detailed analysis
            with st.expander("📊 Detailed Result Analysis", expanded=False):
                st.write(f"**Total results:** {len(outputs)}")
                
                for i, res in enumerate(outputs):
                    st.write(f"\n**Result {i+1}:**")
                    st.write(f"- Status: `{res.get('status', 'unknown')}`")
                    st.write(f"- Valid: `{res.get('valid', 'not set')}`")
                    st.write(f"- Has validated_json: `{'validated_json' in res}`")
                    st.write(f"- Needs user review: `{res.get('needs_user_review', False)}`")
                    st.write(f"- Error: `{res.get('error', 'none')}`")
                    
                    if 'validated_json' in res:
                        st.write(f"- Validated JSON preview: `{str(res['validated_json'])[:100]}...`")
                    elif 'extracted_json' in res:
                        st.write(f"- Extracted JSON preview: `{str(res['extracted_json'])[:100]}...`")
        else:
            st.info("📝 No processing results available. Please run the extraction pipeline first.")

if __name__ == "__main__":
    main() 