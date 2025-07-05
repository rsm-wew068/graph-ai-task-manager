"""
Alternative file upload utilities for handling large files and server restrictions.
Use these when the standard Streamlit file uploader fails with 403 or other errors.
"""

import tempfile
import os
import streamlit as st
from typing import Optional
import io


def create_chunked_upload_interface():
    """
    Create an alternative file upload interface using smaller chunks.
    This can help bypass server restrictions that cause 403 errors.
    """
    st.subheader("🔄 Alternative Upload Method")
    st.info(
        "If normal upload fails with 403 errors, try this chunked upload method. "
        "Split your .mbox file into smaller pieces first."
    )
    
    uploaded_chunks = st.file_uploader(
        "Upload .mbox file chunks (split into <500MB pieces)",
        type=["mbox"],
        accept_multiple_files=True,
        help="Use split command or file splitter to create smaller chunks first"
    )
    
    if uploaded_chunks:
        st.success(f"✅ Uploaded {len(uploaded_chunks)} chunk(s)")
        
        # Combine chunks
        if st.button("Combine and Process Chunks"):
            try:
                combined_file = combine_mbox_chunks(uploaded_chunks)
                st.session_state.uploaded_file = combined_file
                st.success("✅ Chunks combined successfully!")
                return combined_file
            except Exception as e:
                st.error(f"❌ Error combining chunks: {str(e)}")
    
    return None


def combine_mbox_chunks(chunk_files) -> io.BytesIO:
    """
    Combine multiple .mbox chunk files into a single file.
    
    Args:
        chunk_files: List of uploaded file objects
        
    Returns:
        io.BytesIO: Combined file content
    """
    combined_content = io.BytesIO()
    
    # Sort chunks by name to ensure correct order
    sorted_chunks = sorted(chunk_files, key=lambda x: x.name)
    
    for chunk in sorted_chunks:
        chunk.seek(0)
        chunk_data = chunk.read()
        combined_content.write(chunk_data)
    
    combined_content.seek(0)
    return combined_content


def create_local_file_instructions():
    """
    Provide instructions for running the app locally when uploads fail.
    """
    with st.expander("💻 Run Locally (Recommended for Large Files)"):
        st.markdown("""
        **If uploads keep failing, run this app on your computer:**
        
        ```bash
        # Clone the repository
        git clone https://github.com/your-repo/automated-task-manager.git
        cd automated-task-manager
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Run the app locally
        streamlit run app.py
        ```
        
        **Benefits of local installation:**
        - ✅ No file size limits
        - ✅ Faster processing
        - ✅ No network restrictions
        - ✅ Full access to your files
        - ✅ No 403 errors
        
        **System Requirements:**
        - Python 3.8+
        - 4GB+ RAM
        - 2GB+ free disk space
        """)


def create_mbox_splitting_guide():
    """
    Provide instructions for splitting large .mbox files.
    """
    with st.expander("✂️ How to Split Large .mbox Files"):
        st.markdown("""
        **If your .mbox file is too large, split it into smaller pieces:**
        
        **On Windows (PowerShell):**
        ```powershell
        # Split into 500MB chunks
        $file = "Inbox.mbox"
        $chunkSize = 500MB
        $buffer = New-Object byte[] $chunkSize
        $reader = [System.IO.File]::OpenRead($file)
        $count = 0
        while ($reader.Read($buffer, 0, $buffer.Length) -gt 0) {
            [System.IO.File]::WriteAllBytes("inbox_part_$count.mbox", $buffer)
            $count++
        }
        ```
        
        **On Mac/Linux (Terminal):**
        ```bash
        # Split into 500MB chunks
        split -b 500m Inbox.mbox inbox_part_
        
        # Rename with .mbox extension
        for file in inbox_part_*; do
            mv "$file" "$file.mbox"
        done
        ```
        
        **Online Tools:**
        - [File Splitter](https://www.filesplitter.org/)
        - [HJSplit](https://www.hjsplit.org/)
        
        **Then upload the chunks using the alternative upload method above.**
        """)


def validate_upload_environment():
    """
    Check the current environment and provide recommendations.
    """
    st.subheader("🔍 Environment Check")
    
    # Check if running locally vs hosted
    is_local = 'localhost' in st.get_option('browser.serverAddress', '') or \
               st.get_option('server.headless', True) == False
    
    if is_local:
        st.success("✅ Running locally - full file upload capabilities")
    else:
        st.warning("⚠️ Running on hosted platform - may have upload restrictions")
        st.info("💡 For large files, consider running locally for best experience")
    
    # Check available upload size
    max_size = st.get_option('server.maxUploadSize', 200)
    st.info(f"📊 Max upload size configured: {max_size}MB")
    
    return is_local, max_size


def emergency_contact_form():
    """
    Provide a way for users to report persistent upload issues.
    """
    with st.expander("🆘 Still Having Issues? Get Help"):
        st.markdown("""
        **If none of the above solutions work:**
        
        1. **Try different file**: Test with a small sample .mbox file first
        2. **Check file integrity**: Open your .mbox in a text editor
        3. **Network diagnostics**: Try from different internet connection
        4. **Report the issue**: Include these details:
        
        **Debugging Information:**
        """)
        
        # Collect debugging info
        col1, col2 = st.columns(2)
        with col1:
            browser = st.text_input("Browser & Version", placeholder="Chrome 120.0")
            file_size = st.text_input("File Size (MB)", placeholder="1500")
        with col2:
            os_info = st.text_input("Operating System", placeholder="Windows 11")
            error_msg = st.text_area("Error Message", placeholder="Paste the exact error message")
        
        if st.button("📧 Generate Support Email"):
            support_email = f"""
Subject: Upload Error - Automated Task Manager

Browser: {browser}
OS: {os_info}
File Size: {file_size} MB
Error: {error_msg}

Steps already tried:
- [ ] Different browser
- [ ] Smaller file
- [ ] Stable internet
- [ ] File format validation
- [ ] Local installation

Additional context:
[Add any other relevant information]
            """
            st.code(support_email, language='text')
            st.info("📧 Copy this template and email to: support@yourapp.com")
