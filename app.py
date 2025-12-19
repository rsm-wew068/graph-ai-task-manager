import sys
import os
import streamlit as st
import pandas as pd
import json


def check_environment():
    """Check environment variable status for deployment debugging"""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        st.error("""
        🚨 **Missing OPENAI_API_KEY Environment Variable**
        
        The task extraction requires an OpenAI API key to function.
        
        **For Hugging Face Spaces:**
        1. Go to your Space settings
        2. Click on "Repository secrets"
        3. Add `OPENAI_API_KEY` with your OpenAI API key
        4. Restart your Space
        
        **For local development:**
        - Create a `.env` file with `OPENAI_API_KEY=your_key_here`
        - Or set the environment variable in your shell
        """)
        return False
    return True


# Configure Streamlit for large file uploads BEFORE any other operations
try:
    # Set configuration for maximum upload size
    st._config.set_option('server.maxUploadSize', 1024)  # 1GB in MB
except Exception as e:
    # Config setting may fail in some Streamlit versions, continue anyway
    import logging
    logging.debug(f"Could not set maxUploadSize config: {e}")

st.set_page_config(
    page_title="Automated Task Manager", 
    page_icon="📧",
    layout="wide"
)

# Robust path fix for Hugging Face Spaces and local development
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add parent directory for extra safety
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Debug: Print current working directory and Python path
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {current_dir}")
print(f"Python path: {sys.path[:3]}")  # First 3 entries
print(f"Utils directory exists: {os.path.exists(os.path.join(current_dir, 'utils'))}")
print(f"Utils __init__.py exists: {os.path.exists(os.path.join(current_dir, 'utils', '__init__.py'))}")

# Try importing with error handling
try:
    from utils.langgraph_dag import (
        run_extraction_pipeline,
        run_extraction_only_pipeline,
        resume_extraction_pipeline_with_correction
    )
    from utils.embedding import embed_dataframe

    print("✅ Successfully imported utils modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    st.error(f"Import error: {e}")
    st.error("Please ensure all utils files are properly uploaded to your Hugging Face Space")
    st.stop()

# Check environment variables and show warnings if needed
env_status = check_environment()
if not env_status:
    st.warning("⚠️ Application may not work correctly without proper API keys")
    st.info("You can still upload and parse emails, but extraction will fail")

# Add extraction debugging toggle
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🐛 Enable Debug Mode", value=False)
if debug_mode:
    st.sidebar.info("Debug mode will show detailed extraction logs")
    # Store in session state for use during extraction
    st.session_state.debug_mode = True
else:
    st.session_state.debug_mode = False

# Database management buttons
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗃️ Database Management")

# Clear Cached Data button (moved from main area)
if st.sidebar.button("🗑️ Clear Cached Data"):
    for key in ["parsed_emails", "extracted_tasks", "parsing_complete", 
                "processing_complete", "uploaded_file_name", "parse_limit",
                "process_limit", "extracted_graphs"]:
        if key in st.session_state:
            del st.session_state[key]
    st.sidebar.success("✅ Cached data cleared!")
    st.rerun()

# Add Load Emails section with filtering options
st.subheader("📧 Load Emails from Database")

col1, col2, col3 = st.columns(3)
with col1:
    load_limit = st.number_input(
        "Max emails to load",
        min_value=1, 
        max_value=100000, 
        value=1000,
        help="Loading fewer emails is faster"
    )

with col2:
    date_filter = st.selectbox(
        "Date filter",
        ["All dates", "Last 7 days", "Last 30 days", "Last 90 days", "This year", "Most recent first", "Oldest first", "Random sample"],
        help="Filter by email date (for historical data, use 'Most recent first' or 'Random sample')"
    )

with col3:
    sender_filter = st.text_input(
        "Sender filter (optional)",
        placeholder="e.g., @enron.com",
        help="Filter by sender email domain or name"
    )

if st.button("📥 Load Filtered Emails"):
    from utils.database import PostgreSQLDatabase
    import pandas as pd
    from datetime import datetime, timedelta
    
    try:
        db = PostgreSQLDatabase()
        if db.connect():
            # Build the query with filters
            query = "SELECT * FROM parsed_email WHERE 1=1"
            params = []
            
            # Date filter
            if date_filter == "Last 7 days":
                today = datetime.now()
                start_date = today - timedelta(days=7)
                query += " AND date_received >= %s"
                params.append(start_date)
            elif date_filter == "Last 30 days":
                today = datetime.now()
                start_date = today - timedelta(days=30)
                query += " AND date_received >= %s"
                params.append(start_date)
            elif date_filter == "Last 90 days":
                today = datetime.now()
                start_date = today - timedelta(days=90)
                query += " AND date_received >= %s"
                params.append(start_date)
            elif date_filter == "This year":
                today = datetime.now()
                start_date = datetime(today.year, 1, 1)
                query += " AND date_received >= %s"
                params.append(start_date)
            elif date_filter == "Most recent first":
                # No date filter, just order by date descending
                pass
            elif date_filter == "Oldest first":
                # No date filter, just order by date ascending
                pass
            elif date_filter == "Random sample":
                # No date filter, just random order
                pass
            
            # Sender filter
            if sender_filter.strip():
                query += " AND (from_email ILIKE %s OR from_name ILIKE %s)"
                filter_pattern = f"%{sender_filter}%"
                params.extend([filter_pattern, filter_pattern])
            
            # Add ordering and limit
            if date_filter == "Most recent first":
                query += f" ORDER BY date_received DESC LIMIT {load_limit}"
            elif date_filter == "Oldest first":
                query += f" ORDER BY date_received ASC LIMIT {load_limit}"
            elif date_filter == "Random sample":
                query += f" ORDER BY RANDOM() LIMIT {load_limit}"
            else:
                query += f" ORDER BY date_received DESC LIMIT {load_limit}"
            
            # Execute query
            if db.connection is None:
                st.error("Database connection failed")
                return
            with db.connection.cursor() as cursor:
                cursor.execute(query, params)
                emails = [dict(row) for row in cursor.fetchall()]
            
            df_emails = pd.DataFrame(emails)
            st.session_state['parsed_emails'] = df_emails
            st.session_state['parsing_complete'] = True
            st.session_state['uploaded_file_name'] = f'Filtered from PostgreSQL ({len(df_emails)} emails)'
            
            st.success(f"✅ Loaded {len(df_emails)} emails from PostgreSQL database!")
            
            # Show filter summary
            filter_summary = []
            if date_filter != "All dates":
                filter_summary.append(f"Date: {date_filter}")
            if sender_filter.strip():
                filter_summary.append(f"Sender: {sender_filter}")
            if filter_summary:
                st.info(f"📊 Applied filters: {', '.join(filter_summary)}")
            
            db.close()
        else:
            st.error("❌ Failed to connect to PostgreSQL")
    except Exception as e:
        st.error(f"❌ Error loading emails: {str(e)}")

# Quick load options for testing
st.markdown("---")
st.subheader("🚀 Quick Load Options")

col1, col2 = st.columns(2)
with col1:
    if st.button("📧 Load 100 Recent Emails"):
        from utils.database import PostgreSQLDatabase
        import pandas as pd
        
        try:
            db = PostgreSQLDatabase()
            if db.connect():
                with db.connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM parsed_email ORDER BY date_received DESC LIMIT 100")
                    emails = [dict(row) for row in cursor.fetchall()]
                
                df_emails = pd.DataFrame(emails)
                st.session_state['parsed_emails'] = df_emails
                st.session_state['parsing_complete'] = True
                st.session_state['uploaded_file_name'] = 'Recent 100 emails from PostgreSQL'
                
                st.success(f"✅ Loaded {len(df_emails)} recent emails!")
                db.close()
            else:
                st.error("❌ Failed to connect to PostgreSQL")
        except Exception as e:
            st.error(f"❌ Error loading emails: {str(e)}")

with col2:
    if st.button("📧 Load 500 Random Emails"):
        from utils.database import PostgreSQLDatabase
        import pandas as pd
        
        try:
            db = PostgreSQLDatabase()
            if db.connect():
                with db.connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM parsed_email ORDER BY RANDOM() LIMIT 500")
                    emails = [dict(row) for row in cursor.fetchall()]
                
                df_emails = pd.DataFrame(emails)
                st.session_state['parsed_emails'] = df_emails
                st.session_state['parsing_complete'] = True
                st.session_state['uploaded_file_name'] = 'Random 500 emails from PostgreSQL'
                
                st.success(f"✅ Loaded {len(df_emails)} random emails!")
                db.close()
            else:
                st.error("❌ Failed to connect to PostgreSQL")
        except Exception as e:
            st.error(f"❌ Error loading emails: {str(e)}")

# Clear PostgreSQL Database button - only clear tasks (no collaborators table)
if st.sidebar.button("🗑️ Clear PostgreSQL Database"):
    try:
        from utils.database import PostgreSQLDatabase
        db = PostgreSQLDatabase()
        if db.connect():
            with db.connection.cursor() as cursor:
                # Only clear tasks (no collaborators table in new structure)
                cursor.execute("DELETE FROM tasks")
                # Reset sequence to 1
                cursor.execute("ALTER SEQUENCE tasks_id_seq RESTART WITH 1")
            db.close()
            st.sidebar.success("✅ Tasks cleared. Parsed emails remain in database.")
        else:
            st.sidebar.error("❌ Failed to connect to PostgreSQL")
    except Exception as e:
        st.sidebar.error(f"❌ Error clearing PostgreSQL: {e}")

# Clear Neo4j Database button  
if st.sidebar.button("🗑️ Clear Neo4j Database"):
    try:
        from utils.neo4j_graph_writer import Neo4jGraphWriter
        neo4j_writer = Neo4jGraphWriter()
        if neo4j_writer.connect():
            neo4j_writer.clear_graph()
            neo4j_writer.close()
            st.sidebar.success("✅ Neo4j database cleared!")
        else:
            st.sidebar.error("❌ Failed to connect to Neo4j")
    except Exception as e:
        st.sidebar.error(f"❌ Error clearing Neo4j: {e}")


def flatten_extractions(json_list):
    """
    Flatten extracted JSON data into a user-friendly table format.
    
    Args:
        json_list: List of validated JSON objects from extraction (flat structure)
        
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
        
        # Handle flat structure (new format)
        if "task_name" in item:
            rows.append({
                "Task Name": item.get("task_name", "Unnamed Task"),
                "Description": item.get("task_description", ""),
                "Topic": item.get("topic", "Unknown Topic"),
                "Status": item.get("status", "not started"),
                "Priority": item.get("priority_level", "Medium"),
                "Sender": item.get("sender", "Unknown"),
                "Assigned To": item.get("assigned_to", "Unknown"),
                "Due Date": item.get("due_date", ""),
                "Received Date": item.get("received_date", ""),
                "Message ID": item.get("message_id", ""),
                "Spam": item.get("spam", False)
            })
        # Handle old nested structure (for backward compatibility)
        elif "Topic" in item:
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
                collaborators = task.get("collaborators", [])
                if not isinstance(collaborators, list):
                    # If it's a dict, wrap in a list; if not, set to empty list
                    if isinstance(collaborators, dict):
                        collaborators = [collaborators]
                    else:
                        collaborators = []
                if not collaborators:
                    # No collaborators, add a row with 'Unknown' collaborator fields
                    rows.append({
                        "Topic": topic_name,
                        "Task Name": task.get("name", "Unnamed Task"),
                        "Summary": task.get("summary", ""),
                        "Sent Date": task.get("sent_date", ""),
                        "Due Date": task.get("due_date", ""),
                        "Owner Name": owner.get("name", "Unknown") if isinstance(owner, dict) else str(owner) if owner else "Unknown",
                        "Owner Role": owner.get("role", "") if isinstance(owner, dict) else "",
                        "Owner Department": owner.get("department", "") if isinstance(owner, dict) else "",
                        "Owner Organization": owner.get("organization", "") if isinstance(owner, dict) else "",
                        "Collaborator Name": "Unknown",
                        "Collaborator Role": "",
                        "Collaborator Department": "",
                        "Collaborator Organization": "",
                        "Email Index": task_obj.get("email_index", "")
                    })
                else:
                    for collab in collaborators:
                        rows.append({
                            "Topic": topic_name,
                            "Task Name": task.get("name", "Unnamed Task"),
                            "Summary": task.get("summary", ""),
                            "Sent Date": task.get("sent_date", ""),
                            "Due Date": task.get("due_date", ""),
                            "Owner Name": owner.get("name", "Unknown") if isinstance(owner, dict) else str(owner) if owner else "Unknown",
                            "Owner Role": owner.get("role", "") if isinstance(owner, dict) else "",
                            "Owner Department": owner.get("department", "") if isinstance(owner, dict) else "",
                            "Owner Organization": owner.get("organization", "") if isinstance(owner, dict) else "",
                            "Collaborator Name": collab.get("name", "Unknown") if isinstance(collab, dict) else str(collab) if collab else "Unknown",
                            "Collaborator Role": collab.get("role", "") if isinstance(collab, dict) else "",
                            "Collaborator Department": collab.get("department", "") if isinstance(collab, dict) else "",
                            "Collaborator Organization": collab.get("organization", "") if isinstance(collab, dict) else "",
                            "Email Index": task_obj.get("email_index", "")
                        })
    
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="🤖 Automated Task Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📬 Automated Task Manager")

# Updated instructions for the new filtering system
st.markdown('''
### 🚀 Getting Started
1. **Load Emails**: Use the filtering options below to load a manageable subset of emails
2. **Process with LLM**: Extract tasks from the loaded emails
3. **Review & Store**: Validate extracted tasks and store them in databases

**💡 Tip**: Start with 100-1000 emails for faster processing!
''')

# Remove the 'Parse Email' button and demo/sample code
# Update the instructions to reflect the new workflow: Load Emails from PostgreSQL, then Extract Tasks with AI

# Initialize session state variables if not already set
if 'parsing_complete' not in st.session_state:
    st.session_state['parsing_complete'] = False
if 'parsed_emails' not in st.session_state:
    st.session_state['parsed_emails'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'extracted_tasks' not in st.session_state:
    st.session_state['extracted_tasks'] = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = 'enron_sample.parquet'
if 'process_limit' not in st.session_state:
    st.session_state['process_limit'] = 5

# Display parsed emails
if st.session_state.parsing_complete and st.session_state.parsed_emails is not None:
    df_emails = st.session_state.parsed_emails
    
    st.subheader("📊 Parsed Email Data")
    filter_settings = st.session_state.get('filter_settings', {})
    max_limit = filter_settings.get('max_emails_limit', 'unknown')
    st.success(
        f"📊 Parsed {len(df_emails)} emails from "
        f"{st.session_state.uploaded_file_name} "
        f"(filtered from {max_limit} max)"
    )
    
    with st.expander("📊 Email Data Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total emails:** {len(df_emails)}")
            st.write(f"**Columns:** {len(df_emails.columns)}")
            
            # Check for None dates
            none_dates = df_emails['date_received'].isna().sum()
            valid_dates = len(df_emails) - none_dates
            st.write(f"**Valid dates:** {valid_dates}/{len(df_emails)}")
            if none_dates > 0:
                st.warning(f"⚠️ {none_dates} emails have missing dates")
            
            st.write("**Column names:**")
            for col in df_emails.columns:
                st.write(f"  • {col}")
        
        with col2:
            st.write("**Sample emails:**")
            # Display sample emails with correct column names
            if all(col in df_emails.columns for col in ['from_email', 'subject', 'date_received']):
                sample_df = df_emails[['from_email', 'subject', 'date_received']].head(3)
                st.dataframe(sample_df, use_container_width=True)
            else:
                st.warning("Sample columns not found in loaded emails.")
    
    # LLM Processing section
    st.header("🧠 LLM Task Extraction")
    
    col1, col2 = st.columns(2)
    with col1:
        process_limit = st.number_input(
            "Max emails to process with LLM",
            min_value=1, max_value=100000, 
            value=st.session_state.process_limit,
            help="LLM processing is slower and more expensive"
        )
        # Store the limit in session state
        st.session_state.process_limit = process_limit
    
    with col2:
        estimated_time = process_limit * 3  # Rough estimate
        st.info(f"⏱️ Estimated time: ~{estimated_time} seconds")
    
    if st.button("🚀 Start LLM Processing"):
        emails_to_process = df_emails.head(process_limit)
        
        with st.spinner(f"Processing {len(emails_to_process)} emails with LLM..."):
            try:
                # Create embeddings for similarity search
                with st.status("Creating embeddings..."):
                    index, all_chunks = embed_dataframe(emails_to_process)
                
                # Run extraction pipeline for each email
                outputs = []
                progress_bar = st.progress(0)
                
                with st.status("Running LLM extraction...") as status:
                    for i, (_, email_row) in enumerate(emails_to_process.iterrows()):
                        status.update(
                            label=f"Processing email {i+1}/{len(emails_to_process)}: "
                            f"{email_row.get('Subject', 'No Subject')[:50]}..."
                        )
                        
                        # Get the full email row for this iteration
                        email_row = emails_to_process.iloc[i].to_dict()
                        
                        # Get proper email identifier (Message-ID or fallback)
                        message_id = email_row.get('Message-ID')
                        if not message_id:
                            # Fallback to unique identifier if Message-ID missing
                            from_addr = email_row.get('From', 'unknown')
                            subject = email_row.get('Subject', 'no-subject')
                            date = email_row.get('Date', '1970-01-01')
                            base_id = f"{from_addr}_{subject}_{date}"
                            message_id = base_id.replace(' ', '_')[:100000]
                        
                        # Run extraction for this email with full metadata
                        try:
                            print(f"🚀 DEBUG: Starting extraction for email {i+1}")
                            print(f"📧 DEBUG: Subject: {email_row.get('Subject', 'No Subject')}")
                            print(f"📧 DEBUG: Message ID: {message_id}")
                            
                            result = run_extraction_only_pipeline(
                                email_row, index, all_chunks, message_id
                            )
                            
                            print(f"✅ DEBUG: Extraction completed for email {i+1}")
                            print(f"📊 DEBUG: Result keys: {list(result.keys())}")
                            print(f"📊 DEBUG: Status: {result.get('status', 'unknown')}")
                            print(f"📊 DEBUG: Valid: {result.get('valid', 'not set')}")
                            print(f"📊 DEBUG: Has validated_json: {'validated_json' in result}")
                            
                            # Show debug info in Streamlit if enabled
                            if st.session_state.get('debug_mode', False):
                                st.info(f"Email {i+1}: Status={result.get('status')}, Valid={result.get('valid')}")
                            
                            outputs.append(result)
                        except Exception as e:
                            # Handle individual email errors - provide template for HITL
                            email_subject = email_row.get('Subject', 'No Subject')
                            email_content = email_row.get('Body', email_row.get('Content', ''))
                            
                            # Create a helpful template for user validation
                            template_json = {
                                "Topic": {
                                    "name": f"Manual Review: {email_subject[:50]}",
                                    "tasks": [{
                                        "task": {
                                            "name": "Please extract task from email content",
                                            "summary": f"Email content: {email_content[:200]}...",
                                            "sent_date": "",
                                            "due_date": "",
                                            "owner": {
                                                "name": "Unknown",
                                                "role": "Unknown", 
                                                "department": "Unknown",
                                                "organization": "Unknown"
                                            },
                                            "collaborators": []
                                        },
                                        "email_index": message_id
                                    }]
                                }
                            }
                            
                            outputs.append({
                                "email_index": message_id,
                                "status": "error",
                                "error": str(e),
                                "extracted_json": template_json,  # Provide template instead of {}
                                "correctable_json": json.dumps(template_json, indent=2),
                                "valid": False,
                                "needs_user_review": True,  # Trigger HITL
                                "email_content": email_content,  # Show email content for context
                                "email_subject": email_subject
                            })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(emails_to_process))
                
                # Store results in session state
                st.session_state.extracted_tasks = outputs
                st.session_state.processing_complete = True
                
                st.success(
                    f"✅ Completed processing {len(emails_to_process)} emails! "
                    f"Generated {len(outputs)} extraction results."
                )
                
            except Exception as e:
                st.error(f"❌ Error during LLM processing: {str(e)}")

# Display extraction results
if st.session_state.processing_complete and st.session_state.extracted_tasks:
    outputs = st.session_state.extracted_tasks
    
    st.header("📋 Extraction Results (Persistent)")
    
    # Add comprehensive debugging for result categorization
    st.subheader("🔍 Debug: Result Categorization")
    
    with st.expander("📊 Detailed Result Analysis", expanded=False):
        st.write(f"**Total results:** {len(outputs)}")
        
        for i, res in enumerate(outputs):
            st.write(f"\n**Result {i+1}:**")
            st.write(f"- Status: `{res.get('status', 'unknown')}`")
            st.write(f"- Valid: `{res.get('valid', 'not set')}`")
            st.write(f"- Has validated_json: `{'validated_json' in res}`")
            st.write(f"- Has graph: `{'graph' in res}`")
            st.write(f"- Needs user review: `{res.get('needs_user_review', False)}`")
            st.write(f"- Error: `{res.get('error', 'none')}`")
            
            if 'validated_json' in res:
                st.write(f"- Validated JSON preview: `{str(res['validated_json'])[:100000]}...`")
            elif 'extracted_json' in res:
                st.write(f"- Extracted JSON preview: `{str(res['extracted_json'])[:100000]}...`")
    
    # Separate valid and invalid results
    valid_tasks = []
    invalid_results = []
    paused_results = []
    graphs = []
    
    for i, res in enumerate(outputs):
        if "graph" in res:
            graphs.append(res["graph"])
        
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

    # UI feedback after valid_tasks and paused_results are defined
    if valid_tasks:
        st.success(f"✅ Successfully extracted {len(valid_tasks)} valid tasks from {len(outputs)} processed emails.")
    if paused_results:
        st.warning(f"⚠️ {len(paused_results)} emails require manual review. See the Human Validation Required section below.")
    
    # Collect all final tasks (valid, user-validated, and rejected)
    all_final_tasks = valid_tasks.copy()
    if hasattr(st.session_state, 'valid_extraction_results'):
        for validated_result in st.session_state.valid_extraction_results:
            if "validated_json" in validated_result:
                all_final_tasks.append(validated_result["validated_json"])
    # Optionally, add rejected tasks if you track them

    # Graph storage is now handled automatically by Neo4j during processing
    # Each task is stored in Neo4j via the write_graph_node function
    if graphs or valid_tasks:
        st.info(
            f"💾 Task data has been stored in Neo4j graph database and PostgreSQL. "
            f"Processed {len(valid_tasks)} valid tasks for chatbot queries."
        )
    
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
                    help="Edit the JSON to extract the actual task from the email. Use flat structure with fields: task_name, task_description, topic, message_id, status, priority_level, sender, assigned_to, due_date, received_date, spam"
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
                        st.info("� Refreshing page to show updated results...")
                        st.rerun()
                        
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON format: {str(e)}")
                        st.info("💡 Tip: Check for missing commas, quotes, or brackets")
                    except Exception as e:
                        st.error(f"❌ Error during validation: {str(e)}")
                
                if st.button("🚫 Reject", key=f"reject_{idx}"):
                    st.session_state.extracted_tasks[idx].update({
                        "validated_json": parsed_correction,
                        "valid": False,
                        "status": "rejected",
                        "user_corrected_json": corrected_json
                    })
                    st.warning("✅ Task rejected and not added to valid tasks.")
                    st.info(" Refreshing page to show updated results...")
                    st.rerun()
    
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
        
        # Store graphs for persistence
        if graphs:
            st.session_state.extracted_graphs = graphs
        
        # Add Store Data button
        st.markdown("---")
        st.subheader("💾 Store Extracted Data")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("💾 Store to Databases", type="primary"):
                try:
                    # Store valid tasks to PostgreSQL and Neo4j
                    from utils.database import store_validated_tasks
                    from utils.neo4j_graph_writer import Neo4jGraphWriter
                    
                    # Get all valid tasks
                    valid_tasks_to_store = []
                    for task in all_valid_tasks:
                        if isinstance(task, dict) and task.get("task_name"):
                            valid_tasks_to_store.append(task)
                    
                    if valid_tasks_to_store:
                        # Store to PostgreSQL
                        task_ids = store_validated_tasks(valid_tasks_to_store)
                        
                        # Store to Neo4j
                        neo4j_writer = Neo4jGraphWriter()
                        neo4j_success = neo4j_writer.write_tasks_from_table(valid_tasks_to_store, clear_existing=False)
                        
                        st.success(f"✅ Successfully stored {len(task_ids)} tasks to databases!")
                        st.info(f"📊 PostgreSQL: {len(task_ids)} tasks stored")
                        st.info(f"📊 Neo4j: {'Success' if neo4j_success else 'Failed'}")
                        
                        # Mark as stored in session state
                        st.session_state.data_stored = True
                    else:
                        st.warning("⚠️ No valid tasks to store")
                        
                except Exception as e:
                    st.error(f"❌ Error storing data: {str(e)}")
        
        with col2:
            if st.session_state.get('data_stored', False):
                st.success("✅ Data has been stored to databases")
            else:
                st.info("💡 Click 'Store to Databases' to save extracted tasks to PostgreSQL and Neo4j")
    else:
        warning_msg = "⚠️ No valid tasks were extracted from processed emails."
        st.warning(warning_msg)
    
    # Display invalid results (if any remain)
    if invalid_results:
        with st.expander("❌ Invalid/Failed Extractions", expanded=False):
            st.write("These emails failed to extract valid tasks:")
            invalid_df = pd.DataFrame(invalid_results)
            st.dataframe(invalid_df, use_container_width=True)

# After extraction, update UI feedback
# The UI feedback code was moved inside the block where 'valid_tasks' and 'paused_results' are defined.

# Footer
st.markdown("---")
st.markdown('''
📖 **How to use:**
- Click **Parse Email** to load a sample dataset (Enron emails)
- Process with LLM to extract tasks
- Validate any flagged JSON manually (Human-in-the-Loop)
- Explore: Graph visualization, Calendar view, and AI Chatbot
- Data persists across page navigation! 🎉

🔧 **Features:**
- One-click demo: No upload or Gmail Takeout required
- Human-in-the-Loop Validation: Review and correct extracted tasks
- Persistent State: Data stays available across page navigation
- Graph Visualization: See task relationships and dependencies
- Calendar View: Time-based view of extracted tasks
- AI Chatbot: Ask natural language questions about your tasks

🚀 **Tech Stack:**
- LangGraph + GPT: Advanced reasoning pipeline
- FAISS: Vector similarity search for relevant email chunks
- Neo4j: Graph-based task relationship modeling
- Streamlit: Interactive web interface
''')
