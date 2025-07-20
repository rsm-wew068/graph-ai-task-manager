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
from datetime import date, datetime
from utils.langgraph_nodes import convert_dates_to_strings

print("=== APP.PY STARTED ===")

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
            
        # Flat structure only - matches Notion schema exactly
        rows.append({
            "Task Name": item.get("Name", "Unnamed Task"),
            "Task Description": item.get("Task Description", ""),
            "Due Date": item.get("Due Date", ""),
            "Received Date": item.get("Received Date", ""),
            "Status": item.get("Status", ""),
            "Topic": item.get("Topic", ""),
            "Priority Level": item.get("Priority Level", ""),
            "Sender": item.get("Sender", ""),
            "Assigned To": item.get("Assigned To", ""),
            "Email Source": item.get("Email Source", ""),
            "Spam": item.get("Spam", False)
        })
    return pd.DataFrame(rows)

def default_serializer(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)

def main():
    print("=== main() called ===")
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
    print(f"Environment status: {env_status}")
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
        # Check if Neo4j is connected
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
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()["count"]
                st.metric("Total Nodes", node_count)
                st.success("✅ Neo4j Connected")
            driver.close()
        except Exception as e:
            st.metric("Total Nodes", "Not Connected")
            st.error("⚠️ Neo4j connection failed")

        st.markdown("### 🔧 System Info")
        st.info("""
        - **Frontend**: Streamlit ✅
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
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📧 Upload & Process", "📋 Task Management", "🔍 Results Analysis"])
    
    with tab1:
        # Home page content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Welcome to Your AI Task Manager!
            
            This production-ready application helps you manage tasks extracted from your emails using advanced AI capabilities with persistent Neo4j storage.
            """)
            
            # Feature overview
            st.markdown("### 🚀 Key Features")
            
            features = [
                {
                    "title": "📧 Full Email Processing",
                    "description": "Upload Gmail Takeout .mbox files and process entire datasets with no size limits"
                },
                {
                    "title": "🧠 AI-Powered Analysis",
                    "description": "Advanced task extraction, categorization, and priority assessment using LangChain and LangGraph"
                },
                {
                    "title": "🗄️ Persistent Neo4j Storage",
                    "description": "Store tasks, relationships, and metadata in Neo4j with vector indexing for semantic search"
                },
                {
                    "title": "💬 Conversational AI",
                    "description": "Chat with your task manager using natural language queries and commands"
                },
                {
                    "title": "📅 Notion Integration",
                    "description": "Visualize and manage tasks in Notion and sync with your workspace"
                },
                {
                    "title": "👥 Human-in-the-Loop",
                    "description": "Review and validate extracted tasks for better accuracy"
                },
                {
                    "title": "🏭 Production Ready",
                    "description": "Deployed with CI/CD, persistent storage, and enterprise-grade reliability"
                }
            ]
            
            for feature in features:
                with st.container():
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>{feature['title']}</h4>
                        <p>{feature['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📱 Navigation")
            st.markdown("""
            Use the tabs above to navigate:
            - **🏠 Home** - Overview and system status
            - **📧 Upload & Process** - Upload emails and extract tasks
            - **📋 Task Management** - View and manage extracted tasks
            - **🔍 Results Analysis** - Detailed analysis and validation
            
            Or use the sidebar pages:
            - **🤖 AI Chatbot** - Conversational interface
            - **📅 My Calendar** - Visual task calendar view
            """)
    
    with tab2:
        # Upload and Processing Section
        st.markdown("## 📧 Upload & Process Emails")
        
        # Show current data status
        st.subheader("📊 Current Data Status")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.get("parsing_complete", False):
                parsed_emails = st.session_state.get("parsed_emails")
                if parsed_emails is not None and hasattr(parsed_emails, 'empty'):
                    # It's a DataFrame
                    if not parsed_emails.empty:
                        emails_count = len(parsed_emails)
                        st.success(f"✅ {emails_count} emails parsed")
                    else:
                        st.warning("⚠️ Parsing complete but no emails found")
                else:
                    # It's not a DataFrame (shouldn't happen, but safe fallback)
                    st.warning("⚠️ Parsing complete but data format is unexpected")
            else:
                st.info("📁 No emails parsed yet")

        with col2:
            if st.session_state.get("processing_complete", False):
                extracted_tasks = st.session_state.get("extracted_tasks", [])
                task_count = len(extracted_tasks)
                st.success(f"🧠 {task_count} emails processed")
            else:
                st.info("🤖 No LLM processing done")

        with col3:
            extracted_graphs = st.session_state.get("extracted_graphs", [])
            if extracted_graphs:
                graph_count = len(extracted_graphs)
                st.success(f"📊 {graph_count} graphs created")
            else:
                st.info("📈 No graphs generated")
        
        # Upload section
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload Your Gmail Takeout File</h3>
            <p>Upload your Inbox.mbox file from Gmail Takeout to extract tasks using AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Getting Started (Production-Ready Process!)
        1. **Download your Gmail Takeout**: Go to [Google Takeout](https://takeout.google.com/)
           - Select **Mail** → **Multiple formats** → Choose only **"Inbox"**
           - Download as ZIP and **extract/unzip it on your computer**
        2. **Find the Inbox.mbox file**: Look for `Takeout/Mail/Inbox.mbox` in the extracted folder
        3. **Upload the Inbox.mbox file directly** (no size limits!)
        4. **Parse emails** to get a preview of your data
        5. **Run LLM processing** to extract structured tasks
        6. **Use the sidebar** to navigate to Calendar and Chatbot views

        **✨ Production Benefits**: No size limits, persistent Neo4j storage, enterprise-grade reliability!
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your Inbox.mbox file (no size limits)",
            type=['mbox'],
            help="Extract Gmail Takeout ZIP, then upload Inbox.mbox (any size file accepted)."
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Show file upload success and details
            try:
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({file_size:.1f} MB)")
            except Exception as e:
                st.warning(f"⚠️ File uploaded but size check failed: {str(e)}")
            
            # Show helpful info about the production approach
            st.info(
                "📋 **Production file handling**: Upload any size Inbox.mbox file! "
                "No artificial limits - process your entire email archive."
            )
        
        # Filter settings
        st.subheader("🔍 Smart Email Filtering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📅 Date Range**")
            use_date_filter = st.checkbox("Filter by date range", value=True)
            
            if use_date_filter:
                col_start, col_end = st.columns(2)
                with col_start:
                    start_date = st.date_input(
                        "From",
                        value=pd.Timestamp.now() - pd.Timedelta(days=180),
                        help="Parse emails from this date"
                    )
                with col_end:
                    end_date = st.date_input(
                        "To", 
                        value=pd.Timestamp.now(),
                        help="Parse emails up to this date"
                    )
            else:
                start_date = None
                end_date = None
            
        with col2:
            st.markdown("**⚙️ Content Filters**")
            
            keywords = st.text_input(
                "Keywords (optional)",
                placeholder="project, meeting, task, deadline",
                help="Only parse emails containing these keywords"
            )
            
            min_content_length = st.slider(
                "Min content length", 
                min_value=10, max_value=500, value=50,
                help="Skip very short emails"
            )
            
            st.markdown("**🚫 Exclude Types (Suggested)**")
            exclude_types = st.multiselect(
                "Exclude email types",
                ["Notifications", "Newsletters", "Automated"],
                default=["Notifications", "Newsletters"],
                help="Skip common low-value email types"
            )
        
        # Apply Filters button
        if uploaded_file is not None:
            if st.button("🔧 Apply Filters", type="secondary"):
                # Store filter settings in session state
                filter_settings = {
                    "use_date_filter": use_date_filter,
                    "start_date": start_date,
                    "end_date": end_date,
                    "keywords": keywords.split(',') if keywords else [],
                    "min_content_length": min_content_length,
                    "exclude_types": exclude_types
                }
                st.session_state.filter_settings = filter_settings
                st.success("✅ Filters applied! Now you can parse emails.")
                st.rerun()
            
            # Show current filter status
            if hasattr(st.session_state, 'filter_settings'):
                st.info("📋 Filters are configured and ready for parsing")
            else:
                st.warning("⚠️ Please apply filters before parsing emails")
        
        # Processing buttons
        if uploaded_file is not None and hasattr(st.session_state, 'filter_settings'):
            # Use stored filter settings
            filter_settings = st.session_state.filter_settings
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Parse Emails", type="primary"):
                    print("Parse Emails button clicked")
                    try:
                        # Import email parser
                        from utils.email_parser import parse_uploaded_file_with_filters_safe
                        
                        # Parse emails
                        with st.spinner("📧 Parsing emails..."):
                            print("Parsing emails...")
                            emails_df = parse_uploaded_file_with_filters_safe(uploaded_file, filter_settings)
                            print(f"Parsed {len(emails_df)} emails")
                            
                            if not emails_df.empty:
                                st.session_state.parsed_emails = emails_df
                                st.session_state.parsing_complete = True
                                st.success(f"✅ Parsed {len(emails_df)} emails successfully!")
                                
                                # Show preview
                                st.markdown("### 📊 Email Preview")
                                st.dataframe(
                                    emails_df[['Date', 'From', 'Subject', 'content']].head(10),
                                    use_container_width=True
                                )
                            else:
                                st.warning("⚠️ No emails found matching your filters")
                                
                    except Exception as e:
                        print(f"Exception during parsing: {e}")
                        import traceback
                        traceback.print_exc()
                        st.error(f"❌ Error parsing emails: {str(e)}")
            
            with col2:
                if hasattr(st.session_state, 'parsing_complete') and st.session_state.parsing_complete:
                    if st.button("🤖 Extract Tasks with AI", type="primary"):
                        print("=== EXTRACTION PROCESS STARTED ===")
                        print("Extract Tasks with AI button clicked")
                        try:
                            # Import processing modules
                            print("Importing modules...")
                            from utils.langgraph_dag import run_extraction_pipeline
                            from utils.embedding import embed_dataframe
                            print("Modules imported successfully")
                            
                            parsed_emails = st.session_state.parsed_emails
                            print(f"Parsed emails type: {type(parsed_emails)}")
                            print(f"Parsed emails length: {len(parsed_emails) if hasattr(parsed_emails, '__len__') else 'No length'}")
                            
                            if not isinstance(parsed_emails, pd.DataFrame):
                                print("No parsed emails found. Extraction aborted.")
                                st.error("❌ No parsed emails found. Please parse emails before extracting tasks.")
                                return
                            
                            emails_to_process = parsed_emails
                            print(f"Emails to process: {len(emails_to_process)}")
                            
                            with st.spinner("🧠 Extracting tasks with AI..."):
                                print("Creating embeddings...")
                                try:
                                    index, all_chunks = embed_dataframe(emails_to_process)
                                    print(f"Embeddings created successfully. Index: {type(index)}, Chunks: {len(all_chunks)}")
                                except Exception as e:
                                    print(f"Error creating embeddings: {e}")
                                    st.error(f"❌ Error creating embeddings: {str(e)}")
                                    return
                                
                                print("Embeddings created.")
                                outputs = []
                                progress_bar = st.progress(0)
                                
                                with st.status("Running LLM extraction...") as status:
                                    for i, (_, email_row) in enumerate(emails_to_process.iterrows()):
                                        print(f"Processing email {i+1}/{len(emails_to_process)}: {email_row.get('Subject', 'No Subject')}")
                                        status.update(
                                            label=f"Processing email {i+1}/{len(emails_to_process)}: "
                                            f"{email_row.get('Subject', 'No Subject')[:50]}..."
                                        )
                                        # Get the full email row for this iteration
                                        email_row = emails_to_process.iloc[i].to_dict()
                                        print(f"Email row keys: {list(email_row.keys())}")
                                        
                                        # Get proper email identifier (Message-ID or fallback)
                                        message_id = email_row.get('Message-ID')
                                        if not message_id:
                                            from_addr = email_row.get('From', 'unknown')
                                            subject = email_row.get('Subject', 'no-subject')
                                            date = email_row.get('Date', '1970-01-01')
                                            base_id = f"{from_addr}_{subject}_{date}"
                                            message_id = base_id.replace(' ', '_')[:100]
                                        print(f"Message ID: {message_id}")
                                        
                                        # Run extraction for this email with full metadata
                                        try:
                                            print(f"Calling run_extraction_pipeline for email {i+1}...")
                                            result = run_extraction_pipeline(
                                                email_row, index, all_chunks, message_id
                                            )
                                            print(f"Extraction result for email {i+1}: {result.get('status', 'No status')}")
                                            print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                                            
                                            # Write debug output to file
                                            with open("debug_output.txt", "a") as f:
                                                f.write(f"Email {i+1}: {email_row.get('Subject', 'No Subject')}\n")
                                                f.write(f"Extraction result: {json.dumps(result, default=default_serializer, indent=2)}\n\n")
                                            
                                            # Debug print before appending
                                            print("Result to append to outputs:", result)
                                            outputs.append(result)
                                            # Debug print after appending
                                            print("Current outputs list length:", len(outputs))
                                            
                                        except Exception as e:
                                            print(f"Error processing email {i+1}: {e}")
                                            import traceback
                                            traceback.print_exc()
                                            
                                            # Handle individual email errors - provide template for HITL
                                            email_subject = email_row.get('Subject', 'No Subject')
                                            email_content = email_row.get('Body', email_row.get('Content', ''))
                                            # Create a helpful template for user validation
                                            template_json = {
                                                "Name": f"Manual Review: {email_subject[:50]}",
                                                "Task Description": f"Email content: {email_content[:200]}...",
                                                "Due Date": "",
                                                "Received Date": email_row.get("Date", ""),
                                                "Status": "",
                                                "Topic": "",
                                                "Priority Level": "",
                                                "Sender": email_row.get("From", ""),
                                                "Assigned To": "",
                                                "Email Source": message_id,
                                                "Spam": False
                                            }
                                            outputs.append({
                                                "Email Source": message_id,
                                                "status": "error",
                                                "error": str(e),
                                                "extracted_json": template_json,
                                                "correctable_json": json.dumps(convert_dates_to_strings(template_json), indent=2),
                                                "valid": False,
                                                "needs_user_review": True,
                                                "email_content": email_content[:500],
                                                "email_subject": email_subject
                                            })
                                        # Update progress
                                        progress_bar.progress((i + 1) / len(emails_to_process))
                                
                                # Store results in session state
                                print("All outputs before storing in session state:", outputs)
                                st.session_state.extracted_tasks = outputs
                                print("Session state extracted_tasks after storing:", st.session_state.extracted_tasks)
                                st.session_state.processing_complete = True
                                # Count valid tasks
                                valid_tasks = [r for r in outputs if r.get("valid", False)]
                                print(f"Extraction complete. {len(valid_tasks)} valid tasks extracted.")
                                st.success(f"✅ Extracted {len(valid_tasks)} valid tasks!")
                                
                        except Exception as e:
                            print(f"Exception during extraction: {e}")
                            import traceback
                            traceback.print_exc()
                            st.error(f"❌ Error extracting tasks: {str(e)}")
                else:
                    st.info("📝 Parse emails first to enable AI extraction")
    
    with tab3:
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
                        "email_id": res.get("Email Source", "Unknown"),
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
            print("Valid tasks for display:", valid_tasks)
            if hasattr(st.session_state, 'valid_extraction_results'):
                for validated_result in st.session_state.valid_extraction_results:
                    if "validated_json" in validated_result:
                        all_valid_tasks.append(validated_result["validated_json"])
            
            if all_valid_tasks:
                st.subheader("✅ Extracted Valid Tasks")
                
                # Add editing interface for valid tasks
                st.markdown("### ✏️ Edit Tasks")
                st.info("You can edit any task below. Changes will be saved when you click 'Save Changes'.")
                
                # Create editable dataframe
                flattened_df = flatten_extractions(all_valid_tasks)
                
                # Add edit functionality
                edited_df = st.data_editor(
                    flattened_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="task_editor"
                )
                
                # Save changes button
                if st.button("💾 Save Changes", type="primary"):
                    # Convert edited dataframe back to task format
                    updated_tasks = []
                    for _, row in edited_df.iterrows():
                        task = {
                            "Name": row.get("Name", ""),
                            "Task Description": row.get("Task Description", ""),
                            "Due Date": row.get("Due Date", ""),
                            "Received Date": row.get("Received Date", ""),
                            "Status": row.get("Status", ""),
                            "Topic": row.get("Topic", ""),
                            "Priority Level": row.get("Priority Level", ""),
                            "Sender": row.get("Sender", ""),
                            "Assigned To": row.get("Assigned To", ""),
                            "Email Source": row.get("Email Source", ""),
                            "Spam": row.get("Spam", False)
                        }
                        updated_tasks.append(task)
                    
                    # Update session state with edited tasks
                    st.session_state.edited_valid_tasks = updated_tasks
                    st.success("✅ Changes saved! Tasks have been updated.")
                    st.rerun()
                
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
    
    with tab4:
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