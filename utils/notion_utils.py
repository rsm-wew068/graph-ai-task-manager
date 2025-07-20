"""
Notion API utility functions for task management integration.
Requires NOTION_API_KEY and NOTION_DATABASE_ID as environment variables.
"""
import os
from notion_client import Client
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

notion = Client(auth=NOTION_API_KEY) if NOTION_API_KEY else None

def is_configured() -> bool:
    return NOTION_API_KEY is not None and NOTION_DATABASE_ID is not None

def create_task_in_notion(task: Dict[str, Any]) -> Optional[str]:
    """
    Create a new task in the Notion database.
    Expects a dict with keys matching the Notion database properties.
    Returns the created page ID or None on failure.
    """
    if not is_configured():
        raise RuntimeError("Notion API key or database ID not set in environment variables.")
    try:
        properties = {
            "Name": {"title": [{"text": {"content": task.get("Name", "Untitled Task")}}]},
            "Task Description": {"rich_text": [{"text": {"content": task.get("Task Description", "")}}]},
            "Due Date": {"date": {"start": task["Due Date"]}} if task.get("Due Date") else {"date": None},
            "Received Date": {"date": {"start": task["Received Date"]}} if task.get("Received Date") else {"date": None},
            "Status": {"status": {"name": task["Status"]}} if task.get("Status") else {"status": None},
            "Topic": {"select": {"name": task["Topic"]}} if task.get("Topic") else {"select": None},
            "Priority Level": {"select": {"name": task["Priority Level"]}} if task.get("Priority Level") else {"select": None},
            "Sender": {"email": task["Sender"]} if task.get("Sender") else {"email": None},
            "Assigned To": {"email": task["Assigned To"]} if task.get("Assigned To") else {"email": None},
            "Email Source": {"url": task["Email Source"]} if task.get("Email Source") else {"url": None},
            "Spam": {"checkbox": task["Spam"]} if task.get("Spam") else {"checkbox": False}
        }
        response = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties=properties
        )
        return response.get("id")
    except Exception as e:
        print(f"Failed to create Notion task: {e}")
        return None

def query_tasks(filter_: Optional[dict] = None, page_size: int = 10) -> list:
    """
    Query tasks from the Notion database.
    Optionally provide a filter dict as per Notion API.
    """
    if not is_configured():
        raise RuntimeError("Notion API key or database ID not set in environment variables.")
    try:
        # Build query parameters, only include filter if it's not None
        query_params = {
            "database_id": NOTION_DATABASE_ID,
            "page_size": page_size
        }
        
        # Only add filter if it's not None
        if filter_ is not None:
            query_params["filter"] = filter_
        
        result = notion.databases.query(**query_params)
        return result.get("results", [])
    except Exception as e:
        print(f"Failed to query Notion tasks: {e}")
        return []

def clear_notion_database() -> bool:
    """
    Clear all tasks from the Notion database.
    Returns True if successful, False otherwise.
    """
    if not is_configured():
        print("Notion API key or database ID not set in environment variables.")
        return False
    
    try:
        # Get all pages from the database
        all_pages = []
        has_more = True
        start_cursor = None
        
        while has_more:
            query_params = {
                "database_id": NOTION_DATABASE_ID,
                "page_size": 100
            }
            if start_cursor:
                query_params["start_cursor"] = start_cursor
            
            result = notion.databases.query(**query_params)
            pages = result.get("results", [])
            all_pages.extend(pages)
            
            has_more = result.get("has_more", False)
            start_cursor = result.get("next_cursor")
        
        # Delete all pages
        deleted_count = 0
        for page in all_pages:
            try:
                notion.pages.update(page["id"], archived=True)
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete page {page['id']}: {e}")
        
        print(f"Successfully cleared {deleted_count} tasks from Notion database")
        return True
        
    except Exception as e:
        print(f"Failed to clear Notion database: {e}")
        return False
