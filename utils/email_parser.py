# Utils package for automated task manager

import os
import pandas as pd
import re
import mailbox
from email.utils import parsedate_to_datetime
from email.utils import parseaddr


def parse_inbox_mbox(mbox_path: str, max_bytes: int = 200 * 1024 * 1024, max_emails: int = 2000):
    """
    Parse the Inbox.mbox file and yield emails until total read bytes exceeds max_bytes.
    
    Args:
        mbox_path: Path to the Inbox.mbox file
        max_bytes: Maximum bytes to read (default 200MB)
        max_emails: Maximum number of emails to process
        
    Returns:
        pandas.DataFrame: Parsed email data
    """
    if not os.path.exists(mbox_path):
        raise FileNotFoundError(f"Inbox.mbox not found at: {mbox_path}")

    mbox = mailbox.mbox(mbox_path)
    emails = []
    read_bytes = 0
    processed_count = 0

    for msg in mbox:
        # Stop if we've hit our limits
        if processed_count >= max_emails or read_bytes > max_bytes:
            break
            
        try:
            # Calculate message size
            raw_bytes = str(msg).encode("utf-8")
            msg_size = len(raw_bytes)
            
            # Check if adding this message would exceed our limit
            if read_bytes + msg_size > max_bytes:
                break
                
            read_bytes += msg_size

            # Parse date
            date_str = msg.get("Date")
            parsed_date = "1970-01-01T00:00:00"  # Default fallback
            
            if date_str:
                try:
                    parsed_date = parsedate_to_datetime(date_str).isoformat()
                except (ValueError, TypeError):
                    pass

            # Extract names and emails properly
            from_name, from_email = extract_name_and_email(msg.get("From"))
            to_name, to_email = extract_name_and_email(msg.get("To"))
            cc_name, cc_email = extract_name_and_email(msg.get("Cc"))
            bcc_name, bcc_email = extract_name_and_email(msg.get("Bcc"))

            # Get email body
            body = get_text_from_mbox_email(msg)

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
                "content": body,
            }
            emails.append(email_data)
            processed_count += 1

        except Exception as e:
            processed_count += 1
            continue  # Skip problematic emails

    print(f"Parsed {len(emails)} emails (read {read_bytes / (1024*1024):.1f}MB, limit: {max_bytes / (1024*1024):.0f}MB)")
    return pd.DataFrame(emails)

def get_text_from_mbox_email(msg):
    """Extract plain text from mbox email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                try:
                    return part.get_payload(decode=True).decode(
                        part.get_content_charset('utf-8'), errors='replace'
                    )
                except Exception:
                    continue
    else:
        try:
            return msg.get_payload(decode=True).decode(
                msg.get_content_charset('utf-8'), errors='replace'
            )
        except Exception:
            return msg.get_payload()
    return ""


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
        
        # Ensure Date column is datetime (should already be converted)
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Handle timezone awareness - convert both to same timezone or remove timezone
        if df['Date'].dt.tz is not None:
            # If dates are timezone-aware, convert filter dates to UTC
            start_date = start_date.tz_localize('UTC') if start_date.tz is None else start_date
            end_date = end_date.tz_localize('UTC') if end_date.tz is None else end_date
        else:
            # If dates are timezone-naive, remove timezone from filter dates if present
            start_date = start_date.tz_localize(None) if start_date.tz is not None else start_date
            end_date = end_date.tz_localize(None) if end_date.tz is not None else end_date
        
        # Apply date filter directly on Date column
        date_mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df = df[date_mask]
        print(f"📅 Date filter: {len(df)}/{original_count} emails")
    
    # Filter 2: Content length
    min_length = filter_settings.get("min_content_length", 50)
    if min_length > 0:
        df = df.copy()  # Ensure we're working on a copy to avoid warnings
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
            keyword_mask = df['searchable_text'].str.contains(
                '|'.join(keywords), na=False
            )
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
            
            exclude_mask = df['searchable_text'].str.contains(
                '|'.join(exclude_patterns), na=False, case=False
            )
            df = df[~exclude_mask]  # Invert mask to exclude
            print(f"🚫 Excluded {original_count - len(df)} low-value emails")
    
    # Clean up temporary columns
    df = df.drop(columns=[
        col for col in ['Date_parsed', 'content_length', 'searchable_text'] 
        if col in df.columns
    ])
    
    print(f"✅ Final result: {len(df)}/{original_count} emails after filtering")
    return df


def parse_uploaded_file_with_filters_safe(uploaded_file, filter_settings=None):
    """
    Parse uploaded Inbox.mbox file with comprehensive error handling.
    Now expects users to upload the Inbox.mbox file directly (no ZIP).
    Automatically limits processing to first 200MB for any size file.
    """
    if filter_settings is None:
        filter_settings = {}
    
    try:
        # Validate uploaded file
        if uploaded_file is None:
            raise ValueError("No file uploaded")
        
        if not uploaded_file.name.lower().endswith('.mbox'):
            raise ValueError(
                "❌ Please upload an Inbox.mbox file directly.\n\n"
                "Steps to get the file:\n"
                "1. Download your Gmail Takeout ZIP file\n"
                "2. Extract/unzip the file on your computer\n"
                "3. Find and upload the 'Inbox.mbox' file\n"
                "4. The file should be located in: Takeout/Mail/Inbox.mbox"
            )
        
        # Enhanced file access with better error messages
        try:
            uploaded_file.seek(0)
            file_content = uploaded_file.getvalue()
        except Exception as e:
            if "403" in str(e) or "Forbidden" in str(e):
                raise ValueError(
                    "❌ Upload blocked by server (403 error).\n\n"
                    "Solutions to try:\n"
                    "1. Try a smaller .mbox file (< 500MB)\n"
                    "2. Use a different browser (Chrome/Firefox)\n"
                    "3. Check your internet connection\n"
                    "4. Try uploading from a different network\n"
                    "5. Consider running the app locally for large files"
                )
            elif "timeout" in str(e).lower():
                raise ValueError(
                    "❌ Upload timed out.\n\n"
                    "Solutions:\n"
                    "1. Try a smaller file or stable internet connection\n"
                    "2. Split your .mbox file into smaller chunks\n"
                    "3. Use a wired connection instead of WiFi"
                )
            else:
                raise ValueError(f"❌ File upload failed: {str(e)}")
        
        file_size_mb = len(file_content) / (1024 * 1024)
        
        # Validate file content
        if len(file_content) == 0:
            raise ValueError("❌ Uploaded file is empty. Please check your .mbox file.")
        
        # Check if file looks like valid mbox format
        file_start = file_content[:1000].decode('utf-8', errors='ignore')
        if not file_start.startswith('From '):
            raise ValueError(
                "❌ File doesn't appear to be a valid .mbox format.\n\n"
                "Make sure you uploaded the Inbox.mbox file (not a ZIP or other format)."
            )
        
        # Info message about file size handling
        if file_size_mb > 200:
            print(f"📂 File size: {file_size_mb:.1f}MB - processing first 200MB for performance")
        else:
            print(f"📂 File size: {file_size_mb:.1f}MB - processing entire file")
        
        # Save uploaded file temporarily, but only write first 200MB
        import tempfile
        max_bytes_to_write = min(len(file_content), 200 * 1024 * 1024)  # 200MB limit
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp_file:
            # Write only the first 200MB of the file
            tmp_file.write(file_content[:max_bytes_to_write])
            tmp_file.flush()
            
            try:
                # Parse the limited mbox file
                max_emails = filter_settings.get("max_emails_limit", 2000)
                emails_df = parse_inbox_mbox(
                    tmp_file.name, 
                    max_bytes=200 * 1024 * 1024,  # This will process the whole temp file now
                    max_emails=max_emails
                )
                
                # Clean the email content
                if not emails_df.empty:
                    emails_df['content'] = emails_df['content'].apply(clean_email_body)
                    
                    # Convert dates to proper datetime format first
                    emails_df['Date'] = pd.to_datetime(emails_df['Date'], errors='coerce')
                    
                    # Apply filters (which need datetime objects)
                    emails_df = apply_email_filters(emails_df, filter_settings)
                    
                    # Only after filtering, convert to date objects for display
                    # Remove rows with invalid dates first
                    emails_df = emails_df.dropna(subset=['Date'])
                    if not emails_df.empty:
                        emails_df['Date'] = emails_df['Date'].dt.date
                
                return emails_df
                
            finally:
                # Cleanup temp file
                os.unlink(tmp_file.name)
                
    except ValueError:
        # Re-raise ValueError as-is (these are user-friendly messages)
        raise
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            raise ValueError(
                "❌ Server rejected the upload (403 Forbidden).\n\n"
                "This usually means:\n"
                "1. File is too large for the server configuration\n"
                "2. Server security settings are blocking the upload\n"
                "3. Network/proxy restrictions\n\n"
                "Try: smaller file, different browser, or local installation"
            )
        else:
            raise ValueError(f"Email parsing failed: {error_msg}")


def validate_mbox_file_format(file_path):
    """
    Validate that a file is in proper mbox format.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if valid mbox format, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            # Read first few bytes to check format
            header = f.read(1000).decode('utf-8', errors='ignore')
            
            # mbox files should start with "From "
            if not header.startswith('From '):
                return False
                
            # Check for typical email headers
            common_headers = ['Date:', 'From:', 'To:', 'Subject:']
            found_headers = sum(1 for h in common_headers if h in header)
            
            return found_headers >= 2  # At least 2 common headers should be present
            
    except Exception:
        return False
