import sys
import os
import streamlit as st
import pandas as pd
import json

# Configure Streamlit for large file uploads BEFORE any other operations
try:
    # Set configuration for maximum upload size
    st._config.set_option('server.maxUploadSize', 1024)  # 1GB in MB
except Exception:
    pass  # Ignore if config setting fails

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
        resume_extraction_pipeline_with_correction
    )
    from utils.email_parser import parse_uploaded_file_with_filters_safe
    from utils.embedding import embed_dataframe
    from utils.upload_helpers import (
        create_chunked_upload_interface,
        create_local_file_instructions, 
        create_mbox_splitting_guide,
        validate_upload_environment
    )
    print("✅ Successfully imported utils modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    st.error(f"Import error: {e}")
    st.error("Please ensure all utils files are properly uploaded to your Hugging Face Space")
    st.stop()


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


st.set_page_config(
    page_title="🤖 Automated Task Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Streamlit for larger file uploads
try:
    import streamlit.config as config
    config.set_option('server.maxUploadSize', 1000)  # 1GB limit
except Exception:
    pass  # Ignore if config can't be set
st.title("📬 Automated Task Manager")

st.markdown("""
Upload your email archive to extract structured tasks using LangGraph + GPT.  
Then use the sidebar to:
- 🗓 **My Calendar**: View extracted tasks in calendar format
- 🤖 **AI Chatbot**: Ask questions about your tasks

This app supports **Human-in-the-Loop (HITL)** validation for better accuracy.
""")

# Initialize session state for persistence
if "parsed_emails" not in st.session_state:
    st.session_state.parsed_emails = None
if "extracted_tasks" not in st.session_state:
    st.session_state.extracted_tasks = []
if "parsing_complete" not in st.session_state:
    st.session_state.parsing_complete = False
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "process_limit" not in st.session_state:
    st.session_state.process_limit = 5

# Clear data button
if st.button("🗑️ Clear Cached Data"):
    for key in ["parsed_emails", "extracted_tasks", "parsing_complete", 
                "processing_complete", "uploaded_file_name", "parse_limit",
                "process_limit", "extracted_graphs"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ Cached data cleared!")
    st.rerun()

# Show current data status
st.subheader("📊 Current Data Status")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.get("parsing_complete", False):
        emails_count = len(st.session_state.get("parsed_emails", []))
        st.success(f"✅ {emails_count} emails parsed")
    else:
        st.info("📁 No emails parsed yet")

with col2:
    if st.session_state.get("processing_complete", False):
        task_count = len(st.session_state.get("extracted_tasks", []))
        st.success(f"🧠 {task_count} emails processed")
    else:
        st.info("🤖 No LLM processing done")

with col3:
    if st.session_state.get("extracted_graphs", []):
        graph_count = len(st.session_state.get("extracted_graphs", []))
        st.success(f"📊 {graph_count} graphs created")
    else:
        st.info("📈 No graphs generated")

# File upload section
st.header("📂 Upload & Parse Emails")

st.markdown("""
### 🚀 Getting Started (New Simplified Process!)
1. **Download your Gmail Takeout**: Go to [Google Takeout](https://takeout.google.com/)
   - Select **Mail** → **Multiple formats** → Choose only **"Inbox"**
   - Download as ZIP and **extract/unzip it on your computer**
2. **Find the Inbox.mbox file**: Look for `Takeout/Mail/Inbox.mbox` in the extracted folder
3. **Upload the Inbox.mbox file directly** (no ZIP needed!)
4. **Parse emails** to get a preview (first 200MB processed)
5. **Run LLM processing** to extract structured tasks
6. **Use the sidebar** to navigate to Calendar and Chatbot views

**✨ Benefits**: No more ZIP handling = faster, more reliable, no 403 errors!
""")

# Important note about file size warnings
st.warning(
    "⚠️ **Important**: If you see a browser warning about file size (e.g., '200MB limit'), "
    "you can safely ignore it! Our system is configured to handle large files. "
    "Just proceed with uploading your Inbox.mbox file of any size."
)

uploaded_file = st.file_uploader(
    "Upload your Inbox.mbox file (any size - we'll process first 200MB)",
    type=["mbox"],
    help="Extract Gmail Takeout ZIP, then upload Inbox.mbox (any size file accepted)."
)

# Add troubleshooting section for upload errors
with st.expander("🚨 Upload Issues? Click here for solutions"):
    st.markdown("""
    **If you're getting upload errors (403, timeout, etc.):**
    
    1. **File Size**: Try a smaller portion of your .mbox file first
    2. **Browser**: Switch to Chrome or Firefox (better upload handling)
    3. **Connection**: Ensure stable internet during upload
    4. **File Format**: Verify the file ends with `.mbox` (not `.zip`)
    5. **Antivirus**: Temporarily disable antivirus scanning during upload
    
    **Alternative Solutions:**
    - **Split your mbox**: Use a date range export from Gmail (e.g., last 6 months)
    - **Local processing**: Download and run this app locally if upload keeps failing
    - **File validation**: Open your .mbox file in a text editor to ensure it's not corrupted
    
    📖 **[View detailed troubleshooting guide](./TROUBLESHOOTING.md)**
    """)

# Add alternative upload methods for 403 errors
if st.checkbox("🔄 Alternative Upload Methods (for 403 errors)"):
    st.markdown("---")
    
    # Show environment check
    validate_upload_environment()
    
    # Chunked upload option
    create_chunked_upload_interface()
    
    # File splitting guide
    create_mbox_splitting_guide()
    
    # Local installation guide
    create_local_file_instructions()

# Add link to troubleshooting guide
st.sidebar.markdown("---")
st.sidebar.markdown("🆘 **Having upload issues?**")
st.sidebar.markdown("📖 [Troubleshooting Guide](./TROUBLESHOOTING.md)")
st.sidebar.markdown("💡 [Report Issues](https://github.com/your-repo/issues)")

if uploaded_file is not None:
    st.session_state.uploaded_file_name = uploaded_file.name
    
    # Show file upload success and details
    try:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({file_size:.1f} MB)")
    except Exception as e:
        st.warning(f"⚠️ File uploaded but size check failed: {str(e)}")
    
    # Show helpful info about the new parsing approach
    st.info(
        "📋 **Smart file handling**: Upload any size Inbox.mbox file! "
        "We automatically process the first 200MB for optimal performance."
    )
    
    # Intelligent Email Filtering
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
    
    # Advanced options
    with st.expander("🛠️ Advanced Options"):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            max_emails_limit = st.number_input(
                "Max emails (safety limit)",
                min_value=10, max_value=5000, value=2000,
                help="Maximum emails to parse (we auto-limit to first 200MB of file)"
            )
        with col_adv2:
            exclude_types = st.multiselect(
                "Exclude email types",
                ["Notifications", "Newsletters", "Automated"],
                help="Skip common low-value email types"
            )
    
    # Store filter settings in session state
    filter_settings = {
        "use_date_filter": use_date_filter,
        "start_date": start_date,
        "end_date": end_date,
        "keywords": keywords.split(',') if keywords else [],
        "min_content_length": min_content_length,
        "max_emails_limit": max_emails_limit,
        "exclude_types": exclude_types
    }
    st.session_state.filter_settings = filter_settings
    
    # Parse emails button
    if st.button("📊 Parse Emails with Filters"):
        with st.spinner("Parsing emails with smart filters..."):
            try:
                df_emails = parse_uploaded_file_with_filters_safe(
                    uploaded_file, filter_settings
                )
                st.session_state.parsed_emails = df_emails
                st.session_state.parsing_complete = True
                st.success(f"✅ Successfully parsed {len(df_emails)} emails!")
            except Exception as e:
                st.error(f"❌ Error parsing emails: {str(e)}")

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
            none_dates = df_emails['Date'].isna().sum()
            valid_dates = len(df_emails) - none_dates
            st.write(f"**Valid dates:** {valid_dates}/{len(df_emails)}")
            if none_dates > 0:
                st.warning(f"⚠️ {none_dates} emails have missing dates")
            
            st.write("**Column names:**")
            for col in df_emails.columns:
                st.write(f"  • {col}")
        
        with col2:
            st.write("**Sample emails:**")
            sample_df = df_emails[['From', 'Subject', 'Date']].head(3)
            st.dataframe(sample_df, use_container_width=True)
    
    # LLM Processing section
    st.header("🧠 LLM Task Extraction")
    
    col1, col2 = st.columns(2)
    with col1:
        process_limit = st.number_input(
            "Max emails to process with LLM",
            min_value=1, max_value=100, 
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
                            message_id = base_id.replace(' ', '_')[:100]
                        
                        # Run extraction for this email with full metadata
                        try:
                            result = run_extraction_pipeline(
                                email_row, index, all_chunks, message_id
                            )
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
                                            "start_date": "",
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
                                "email_content": email_content[:500],  # Show email content for context
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
            res.get("needs_user_review", False) or
            res.get("needs_human_review", False)):
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
    
    # Combine all graphs into one master graph for chatbot queries
    if graphs:
        import networkx as nx
        import pickle
        
        # Create a combined graph
        master_graph = nx.DiGraph()
        for graph in graphs:
            if graph:  # Only process non-None graphs
                master_graph = nx.compose(master_graph, graph)
        
        # Save the combined graph for chatbot queries
        with open("topic_graph.gpickle", "wb") as f:
            pickle.dump(master_graph, f)
        
        st.info(
            f"💾 Saved combined knowledge graph with "
            f"{len(master_graph.nodes())} nodes for chatbot queries"
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
                        st.info("� Refreshing page to show updated results...")
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
        
        # Store graphs for persistence
        if graphs:
            st.session_state.extracted_graphs = graphs
    else:
        warning_msg = "⚠️ No valid tasks were extracted from processed emails."
        st.warning(warning_msg)
    
    # Display invalid results (if any remain)
    if invalid_results:
        with st.expander("❌ Invalid/Failed Extractions", expanded=False):
            st.write("These emails failed to extract valid tasks:")
            invalid_df = pd.DataFrame(invalid_results)
            st.dataframe(invalid_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### 📖 How to use:
1. **Upload** your Inbox.mbox file (extracted from Gmail Takeout)
2. **Parse** emails (first 200MB processed for performance) 
3. **Process** with LLM (slower, extracts tasks)
4. **Validate** any flagged JSON manually
5. **Navigate** to other pages to view results

Data persists across page navigation! 🎉

### 🔧 Features:
- **Direct .mbox Upload**: No ZIP handling, faster and more reliable
- **Memory-Safe**: Processes first 200MB to avoid memory issues 
- **Human-in-the-Loop Validation**: Review and correct extracted JSON
- **Persistent State**: Data stays available across page navigation
- **Graph Visualization**: See task relationships and dependencies
- **Calendar View**: Time-based view of extracted tasks
- **AI Chatbot**: Ask natural language questions about your tasks

### 🚀 Tech Stack:
- **LangGraph + GPT**: Advanced reasoning pipeline
- **FAISS**: Vector similarity search for relevant email chunks
- **NetworkX**: Graph-based task relationship modeling
- **Streamlit**: Interactive web interface
""")
