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

def parse_mbox(mbox_path):
    mbox = mailbox.mbox(mbox_path)
    emails = []
    for msg in mbox:
        try:
            email_data = {
                "Message-ID": msg.get("Message-ID"),
                "Date": parsedate_to_datetime(msg.get("Date")).isoformat() if msg.get("Date") else None,
                "From": msg.get("From"),
                "To": msg.get("To"),
                "Subject": msg.get("Subject"),
                "content": get_text_from_mbox_email(msg),
                "X-From": msg.get("From"),
                "X-To": msg.get("To"),
                "X-cc": msg.get("Cc"),
                "X-bcc": msg.get("Bcc"),
            }
            emails.append(email_data)
        except Exception:
            continue
    return pd.DataFrame(emails)

def extract_and_find_mbox(zip_path, extract_to="extracted_mbox"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".mbox"):
                return os.path.join(root, file)
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
    """Extract display name from email address string."""
    if not isinstance(x, str):
        return ""
    match = re.match(r"^([^<@]+)", x)
    if match:
        name = match.group(1).strip()
        # If it's "Last, First", flip it
        if ',' in name:
            parts = [p.strip() for p in name.split(',')]
            if len(parts) == 2:
                return f"{parts[1]} {parts[0]}"
        return name
    return x.strip()


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
    # Step 1: Clean timezone labels
    df_emails['clean_date'] = df_emails['Date'].str.replace(
        r'\s+\(.*\)', '', regex=True)

    # Step 2: Parse datetime safely & force conversion
    df_emails['Date'] = pd.to_datetime(
        df_emails['clean_date'], errors='coerce', utc=True)

    # Drop columns that exist
    columns_to_drop = ['Mime-Version', 'Content-Type', 'Content-Transfer-Encoding', 'clean_date']
    existing_columns = [col for col in columns_to_drop if col in df_emails.columns]
    if existing_columns:
        df_emails = df_emails.drop(columns=existing_columns)

    # Extract display names (only if columns exist)
    if 'X-From' in df_emails.columns:
        df_emails['Name-From'] = df_emails['X-From'].apply(extract_display_name)
    if 'X-To' in df_emails.columns:
        df_emails['Name-To'] = df_emails['X-To'].apply(extract_display_name)
    if 'X-cc' in df_emails.columns:
        df_emails['Name-cc'] = df_emails['X-cc'].apply(extract_display_name)
    if 'X-bcc' in df_emails.columns:
        df_emails['Name-bcc'] = df_emails['X-bcc'].apply(extract_display_name)

    # Clean email content
    df_emails['content'] = df_emails['content'].apply(clean_email_body)

    # Drop X-columns that exist
    x_columns_to_drop = ['X-From', 'X-To', 'X-cc', 'X-bcc']
    existing_x_columns = [col for col in x_columns_to_drop if col in df_emails.columns]
    if existing_x_columns:
        df_emails = df_emails.drop(columns=existing_x_columns)

    # Convert date to date only
    df_emails['Date'] = pd.to_datetime(df_emails['Date'])
    df_emails['Date'] = df_emails['Date'].dt.date

    return df_emails

def parse_uploaded_file(uploaded_file):
    """Parse an uploaded ZIP file containing .mbox data."""
    import tempfile
    import os
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract and find mbox file
        mbox_path = extract_and_find_mbox(tmp_path)
        
        if not mbox_path:
            raise ValueError("No .mbox file found in the uploaded ZIP")
        
        # Parse the mbox file
        df_emails = parse_mbox(mbox_path)
        
        # Clean the dataframe
        df_emails = clean_dataframe(df_emails)
        
        return df_emails
    
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
