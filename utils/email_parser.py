# %% [markdown]
# # Load Data

# %%
import os
import pandas as pd
import re
# --- Gmail Takeout .mbox loader fallback ---
import mailbox
from email.utils import parsedate_to_datetime
import zipfile
from email.utils import parseaddr

def get_text_from_mbox_email(msg):
    """Extract plain text from mbox email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset('utf-8'), errors='replace')
                except:
                    continue
    else:
        try:
            return msg.get_payload(decode=True).decode(msg.get_content_charset('utf-8'), errors='replace')
        except:
            return msg.get_payload()
    return ""

def parse_mbox(mbox_path, max_emails=None):
    """Parse mbox file with early termination at limit."""
    mbox = mailbox.mbox(mbox_path)
    emails = []
    processed_count = 0
    
    # Set a reasonable default if no limit specified
    if max_emails is None:
        max_emails = 100  # Default limit to prevent huge processing
    
    for msg in mbox:
        # Stop immediately when we hit the limit
        if processed_count >= max_emails:
            break
            
        try:
            # Simplified date parsing - less fallback for speed
            date_str = msg.get("Date")
            parsed_date = "1970-01-01T00:00:00"  # Default fallback
            
            if date_str:
                try:
                    parsed_date = parsedate_to_datetime(date_str).isoformat()
                except (ValueError, TypeError):
                    # Skip complex fallback parsing for speed
                    pass
            # Extract names and emails properly
            from_name, from_email = extract_name_and_email(msg.get("From"))
            to_name, to_email = extract_name_and_email(msg.get("To"))
            cc_name, cc_email = extract_name_and_email(msg.get("Cc"))
            bcc_name, bcc_email = extract_name_and_email(msg.get("Bcc"))
            
            email_data = {
                "Message-ID": msg.get("Message-ID"),
                "Date": parsed_date,
                "From": from_email,
                "To": to_email,
                "Cc": cc_email,
                "Bcc": bcc_email,
                "Name-From": from_name,
                "Name-To": to_name,
                "Name-Cc": cc_name,
                "Name-Bcc": bcc_name,
                "Subject": msg.get("Subject"),
                "content": get_text_from_mbox_email(msg),
            }
            emails.append(email_data)
            processed_count += 1
            
        except Exception:
            # Still count failed emails toward the limit to avoid infinite loops
            processed_count += 1
            continue
    
    print(f"Parsed {len(emails)} emails (limit: {max_emails})")
    return pd.DataFrame(emails)

def extract_and_find_mbox(zip_path, extract_to="extracted_mbox"):
    """Find mbox file in ZIP without extracting everything."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # First, just list files to find .mbox without extracting
        for file_info in zip_ref.filelist:
            if file_info.filename.endswith(".mbox"):
                # Extract only the .mbox file
                zip_ref.extract(file_info.filename, extract_to)
                return os.path.join(extract_to, file_info.filename)
    
    raise FileNotFoundError("No .mbox file found in uploaded Gmail ZIP.")

# Example usage (only run when script is executed directly)
if __name__ == "__main__":
    # Use Gmail .zip archive if present, else raise error
    zip_path = "takeout.zip"
    if os.path.exists(zip_path):
        extracted_mbox_path = extract_and_find_mbox(zip_path)
        df_emails = parse_mbox(extracted_mbox_path)
        print(df_emails.shape)
        print(df_emails.head())
    else:
        print(f"No takeout.zip file found at {zip_path}")
        print("Use the functions extract_and_find_mbox() and parse_mbox() "
              "directly with your ZIP file path.")

# %% [markdown]
# # Clean Data


def extract_display_name(x):
    """Extract and clean display name from email address string."""
    if not isinstance(x, str):
        return ""
    
    # Handle multiple email addresses separated by commas
    if ',' in x and '@' in x.split(',')[0] and '@' in x.split(',')[1]:
        # Multiple emails - take the first one
        x = x.split(',')[0].strip()
    
    # Extract name part before < or @
    match = re.match(r"^([^<@]+)", x)
    if match:
        name = match.group(1).strip()
    else:
        # Fallback: try to extract from angle brackets
        angle_match = re.search(r'"?([^"<>]+)"?\s*<', x)
        if angle_match:
            name = angle_match.group(1).strip()
        else:
            name = x.strip()
    
    # Clean formatting symbols and quotes
    name = re.sub(r'^["\']|["\']$', '', name)  # Remove surrounding quotes
    name = re.sub(r'[<>]', '', name)  # Remove angle brackets
    # Clean extra quotes/spaces
    name = re.sub(r'^\s*["\']*\s*|\s*["\']*\s*$', '', name)
    
    # Keep letters, spaces, hyphens, periods, commas, apostrophes
    name = re.sub(r'[^\w\s\-\.,\']+', '', name)
    name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
    name = name.strip()
    
    # Handle "Last, First" format
    if ',' in name and '@' not in name:
        parts = [p.strip() for p in name.split(',')]
        if len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0:
            # Only flip if both parts look like names (not email-like)
            part0_clean = not re.search(r'[@\.]', parts[0])
            part1_clean = not re.search(r'[@\.]', parts[1])
            if part0_clean and part1_clean:
                return f"{parts[1]} {parts[0]}"
    
    # If result is empty or looks like an email, try to extract from original
    if not name or '@' in name:
        # Try to extract from email address username
        email_match = re.search(r'([^@<>\s]+)@', x)
        if email_match:
            username = email_match.group(1)
            # Convert common username patterns to readable names
            username = re.sub(r'[._-]', ' ', username)
            username = username.title()
            return username if username else "Unknown"
    
    return name if name else "Unknown"


def clean_email_body(text):
    """Clean email body text by removing line breaks and normalizing."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ')  # remove line breaks
    text = text.replace('\t', ' ')  # remove tabs
    text = re.sub(r'\s+', ' ', text)  # normalize extra whitespace
    # optional: remove weird characters
    text = re.sub(r'[^a-zA-Z0-9.,!?$%:;/@#\'\"()\- ]', '', text)
    return text.strip()


def clean_dataframe(df_emails):
    """Clean the email dataframe with various transformations."""
    # Step 1: Filter out emails with None dates first
    df_emails = df_emails.dropna(subset=['Date'])
    
    # Step 2: Clean timezone labels (only for string dates)
    if df_emails['Date'].dtype == 'object':
        df_emails['clean_date'] = df_emails['Date'].str.replace(
            r'\s+\(.*\)', '', regex=True)
    else:
        df_emails['clean_date'] = df_emails['Date']

    # Step 3: Parse datetime safely & force conversion
    df_emails['Date'] = pd.to_datetime(
        df_emails['clean_date'], errors='coerce', utc=True)
    
    # Step 4: Drop any rows where date parsing failed
    df_emails = df_emails.dropna(subset=['Date'])

    # Drop columns that exist
    columns_to_drop = ['Mime-Version', 'Content-Type', 'Content-Transfer-Encoding', 'clean_date']
    existing_columns = [col for col in columns_to_drop if col in df_emails.columns]
    if existing_columns:
        df_emails = df_emails.drop(columns=existing_columns)

    # Names are already extracted during parsing, no need for additional processing
    print(f"✅ Email processing complete: {len(df_emails)} emails with names extracted")

    # Clean email content
    df_emails['content'] = df_emails['content'].apply(clean_email_body)

    # Convert date to date only
    df_emails['Date'] = pd.to_datetime(df_emails['Date'])
    df_emails['Date'] = df_emails['Date'].dt.date

    return df_emails

def parse_uploaded_file(uploaded_file, max_emails=None):
    """Parse an uploaded ZIP file containing .mbox data efficiently."""
    import tempfile
    import os
    import io
    
    # For small files, process in memory; for large files, use temp file
    file_size = len(uploaded_file.getvalue())
    
    if file_size < 50 * 1024 * 1024:  # Less than 50MB - process in memory
        try:
            # Parse directly from memory
            zip_bytes = io.BytesIO(uploaded_file.getvalue())
            with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
                # Find .mbox file
                mbox_filename = None
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith(".mbox"):
                        mbox_filename = file_info.filename
                        break
                
                if not mbox_filename:
                    raise ValueError("No .mbox file found in the uploaded ZIP")
                
                # Extract .mbox content to memory
                with zip_ref.open(mbox_filename) as mbox_file:
                    # Create temporary file for mbox content only
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix='.mbox'
                    ) as tmp_mbox:
                        tmp_mbox.write(mbox_file.read())
                        tmp_mbox_path = tmp_mbox.name
                
                try:
                    # Parse the mbox file with optional limit
                    df_emails = parse_mbox(
                        tmp_mbox_path, max_emails=max_emails
                    )
                    
                    # Clean the dataframe
                    df_emails = clean_dataframe(df_emails)
                    
                    return df_emails
                finally:
                    # Cleanup temporary mbox file
                    if os.path.exists(tmp_mbox_path):
                        os.unlink(tmp_mbox_path)
                        
        except Exception:
            # Silently fall back to disk processing for large files
            pass
    
    # Fallback: Large files or if memory processing fails
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract and find mbox file (now optimized)
        mbox_path = extract_and_find_mbox(tmp_path)
        
        if not mbox_path:
            raise ValueError("No .mbox file found in the uploaded ZIP")
        
        # Parse the mbox file with optional limit
        df_emails = parse_mbox(mbox_path, max_emails=max_emails)
        
        # Clean the dataframe
        df_emails = clean_dataframe(df_emails)
        
        return df_emails
    
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        # Cleanup extracted directory
        extract_dir = "extracted_mbox"
        if os.path.exists(extract_dir):
            import shutil
            shutil.rmtree(extract_dir)

def parse_uploaded_file_with_filters(uploaded_file, filter_settings):
    """
    Parse uploaded ZIP file with intelligent filtering options.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        filter_settings: Dict with filtering options
    
    Returns:
        pandas.DataFrame: Filtered email data
    """
    import tempfile
    import os
    import io
    from datetime import datetime
    
    # For small files, process in memory; for large files, use temp file
    file_size = len(uploaded_file.getvalue())
    
    # Step 1: Extract and parse all emails first
    if file_size < 50 * 1024 * 1024:  # Less than 50MB
        try:
            # Parse directly from memory
            bytes_data = uploaded_file.getvalue()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(bytes_data)
                tmp_path = tmp_file.name
            
            # Extract and find mbox
            mbox_path = extract_and_find_mbox(tmp_path, "extracted_mbox")
            
            if not mbox_path:
                return pd.DataFrame()
            
            # Parse with high limit first, then filter
            df_all = parse_mbox(mbox_path, max_emails=filter_settings.get("max_emails_limit", 500))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Large file - use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            mbox_path = extract_and_find_mbox(tmp_path, "extracted_mbox")
            if not mbox_path:
                return pd.DataFrame()
            
            df_all = parse_mbox(mbox_path, max_emails=filter_settings.get("max_emails_limit", 500))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Step 2: Apply intelligent filters
    df_filtered = apply_email_filters(df_all, filter_settings)
    
    # Cleanup
    extract_dir = "extracted_mbox"
    if os.path.exists(extract_dir):
        import shutil
        shutil.rmtree(extract_dir)
    
    return df_filtered


def apply_email_filters(df, filter_settings):
    """
    Apply intelligent filters to email DataFrame.
    
    Args:
        df: DataFrame with parsed emails
        filter_settings: Dict with filter options
        
    Returns:
        pandas.DataFrame: Filtered emails
    """
    if df.empty:
        return df
    
    original_count = len(df)
    
    # Filter 1: Date range
    if filter_settings.get("use_date_filter") and filter_settings.get("start_date"):
        start_date = pd.to_datetime(filter_settings["start_date"])
        end_date = pd.to_datetime(filter_settings["end_date"])
        
        # Convert email dates to datetime
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Apply date filter
        date_mask = (df['Date_parsed'] >= start_date) & (df['Date_parsed'] <= end_date)
        df = df[date_mask]
        print(f"📅 Date filter: {len(df)}/{original_count} emails")
    
    # Filter 2: Content length
    min_length = filter_settings.get("min_content_length", 50)
    if min_length > 0:
        df['content_length'] = df['content'].fillna('').str.len()
        df = df[df['content_length'] >= min_length]
        print(f"📝 Content filter: {len(df)} emails with >{min_length} chars")
    
    # Filter 3: Keywords
    keywords = filter_settings.get("keywords", [])
    if keywords:
        keywords = [k.strip().lower() for k in keywords if k.strip()]
        if keywords:
            # Create combined text for searching
            df['searchable_text'] = (
                df['Subject'].fillna('') + ' ' + 
                df['content'].fillna('')
            ).str.lower()
            
            # Check if any keyword is present
            keyword_mask = df['searchable_text'].str.contains('|'.join(keywords), na=False)
            df = df[keyword_mask]
            print(f"🔍 Keyword filter: {len(df)} emails with keywords: {keywords}")
    
    # Filter 4: Exclude common low-value emails
    exclude_types = filter_settings.get("exclude_types", [])
    if exclude_types:
        exclude_patterns = []
        if "Notifications" in exclude_types:
            exclude_patterns.extend(['notification', 'alert', 'reminder'])
        if "Newsletters" in exclude_types:
            exclude_patterns.extend(['newsletter', 'unsubscribe', 'marketing'])
        if "Automated" in exclude_types:
            exclude_patterns.extend(['noreply', 'no-reply', 'automated', 'system'])
        
        if exclude_patterns:
            # Create searchable text from subject, from, and content
            df['searchable_text'] = (
                df['Subject'].fillna('') + ' ' + 
                df['From'].fillna('') + ' ' + 
                df['content'].fillna('')
            ).str.lower()
            
            exclude_mask = df['searchable_text'].str.contains('|'.join(exclude_patterns), na=False, case=False)
            df = df[~exclude_mask]  # Invert mask to exclude
            print(f"🚫 Excluded {original_count - len(df)} low-value emails")
    
    # Clean up temporary columns
    df = df.drop(columns=[col for col in ['Date_parsed', 'content_length', 'searchable_text'] 
                         if col in df.columns])
    
    print(f"✅ Final result: {len(df)}/{original_count} emails after filtering")
    return df

def extract_name_and_email(email_string):
    """
    Extract display name and email address from email header.
    
    Args:
        email_string: Raw email header like 'John Doe <john@example.com>'
        
    Returns:
        tuple: (display_name, email_address)
    """
    if not email_string:
        return "", ""
    
    try:
        # Use email.utils.parseaddr for proper parsing
        name, email = parseaddr(email_string)
        
        # Clean up the name - remove quotes and extra whitespace
        if name:
            name = name.strip('"').strip("'").strip()
        
        # If no name was found, try to extract from email
        if not name and email:
            # Try to get name from email prefix (before @)
            local_part = email.split('@')[0] if '@' in email else email
            # Convert dots/underscores to spaces and title case
            name = local_part.replace('.', ' ').replace('_', ' ').title()
        
        return name or "Unknown", email or ""
        
    except Exception:
        # Fallback - return the original string as email
        return "Unknown", email_string or ""
