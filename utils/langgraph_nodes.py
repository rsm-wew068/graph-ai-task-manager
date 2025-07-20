import re
import os
from datetime import datetime, date
from langchain_openai import ChatOpenAI
import requests
import re
from utils.notion_utils import create_task_in_notion
from utils.prompt_template import rag_extraction_prompt, reason_prompt
from utils.embedding import retrieve_similar_chunks
from utils.graph_writer import write_tasks_to_neo4j
import time

def convert_dates_to_strings(obj):
    if isinstance(obj, dict):
        return {k: convert_dates_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates_to_strings(i) for i in obj]
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    else:
        return obj

# Node: write to Notion database
def write_notion_node(state):
    """
    Insert the extracted/validated task into the Notion database using the exact field names from the schema.
    """
    print("=== WRITE_NOTION_NODE STARTED ===")
    extracted = state["validated_json"]
    print(f"Extracted task: {extracted}")
    
    notion_task = {
        "Name": extracted.get("Name", "Unnamed Task"),
        "Task Description": extracted.get("Task Description", ""),
        "Due Date": extracted.get("Due Date"),
        "Received Date": extracted.get("Received Date"),
        "Status": extracted.get("Status", "Not started"),
        "Topic": extracted.get("Topic", ""),
        "Priority Level": extracted.get("Priority Level", ""),
        "Sender": extracted.get("Sender", ""),
        "Assigned To": extracted.get("Assigned To", ""),
        "Email Source": extracted.get("Email Source", ""),
        "Spam": extracted.get("Spam", False)
    }
    print(f"Notion task prepared: {notion_task}")
    
    notion_id = create_task_in_notion(notion_task)
    print(f"Notion task created with ID: {notion_id}")
    
    return {"notion_status": "success" if notion_id else "error", "notion_id": notion_id, **state}

# Handle dotenv import gracefully for deployment environments
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set directly
    pass



# Get OpenAI API key with validation
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("❌ WARNING: OPENAI_API_KEY environment variable not found!")
    print("Available environment variables:")
    for key in sorted(os.environ.keys()):
        if 'API' in key.upper() or 'KEY' in key.upper():
            value_preview = ('***' if key.endswith('_KEY')
                             else os.environ[key][:20] + '...')
            print(f"  {key}: {value_preview}")
else:
    print(f"✅ OPENAI_API_KEY found: {openai_key[:10]}...")


def get_llm():
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required but not set. "
            "Please set it in your environment variables or .env file."
        )
    return ChatOpenAI(model="gpt-4o", temperature=0.2,
                      openai_api_key=openai_key)


# Node: chunked prompt creation for RAG using full email metadata
def rag_prompt_node(state):
    # Get the current email row data
    current_email = convert_dates_to_strings(state.get("current_email_row", {}))
    
    # Format the main email with all metadata for LLM analysis
    main_email_context = format_email_for_llm(current_email)
    
    # Use content for similarity search but provide full context to LLM
    query_text = current_email.get("content", "")
    if query_text and state.get("faiss_index") is not None:
        related = retrieve_similar_chunks(
            query_text,
            index=state["faiss_index"],
            chunks=state["all_chunks"],
            k=2
        )
        
        # Related emails are just text chunks - we only have full metadata
        # for current email
        related_email_1 = related[0] if len(related) > 0 else ""
        related_email_2 = related[1] if len(related) > 1 else ""
    else:
        related_email_1 = ""
        related_email_2 = ""

    prompt = rag_extraction_prompt.format(
        main_email=main_email_context,
        related_email_1=related_email_1,
        related_email_2=related_email_2
    )
    return {"rag_prompt": prompt, **state}


def format_email_for_llm(email_row):
    """
    Format a complete email row with all metadata for LLM analysis.
    
    Args:
        email_row: Dictionary containing all email fields from parser
        
    Returns:
        Formatted string with all email metadata and content
    """
    if not email_row:
        return "No email data available"
    
    # Convert all date/datetime fields to strings
    email_row = convert_dates_to_strings(email_row)

    # Extract all available fields
    message_id = email_row.get("Message-ID", "Unknown")
    date = email_row.get("Date", "Unknown")
    from_email = email_row.get("From", "Unknown")
    to_email = email_row.get("To", "Unknown")
    cc_email = email_row.get("Cc", "") or "None"
    bcc_email = email_row.get("Bcc", "") or "None"
    from_name = email_row.get("Name-From", "Unknown")
    to_name = email_row.get("Name-To", "Unknown")
    cc_name = email_row.get("Name-Cc", "") or "None"
    bcc_name = email_row.get("Name-Bcc", "") or "None"
    subject = email_row.get("Subject", "No Subject")
    content = email_row.get("content", "No content")
    
    formatted = f"""EMAIL METADATA:
Message-ID: {message_id}
Date: {date}
From: {from_name} <{from_email}>
To: {to_name} <{to_email}>
Cc: {cc_name} <{cc_email}>
Bcc: {bcc_name} <{bcc_email}>
Subject: {subject}

EMAIL CONTENT:
{content}
"""
    
    return formatted


# Node: GPT call to extract task JSON
def extract_json_node(state):
    llm = get_llm()
    result = llm.invoke(state["rag_prompt"])
    raw = result.content.strip()

    # Auto-strip common GPT wrapping
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    # Auto-fix common JSON issues
    cleaned = auto_fix_json(raw)

    # Write raw and cleaned LLM output to debug_output.txt
    with open("debug_output.txt", "a") as f:
        f.write("=== RAW LLM OUTPUT ===\n")
        f.write(raw + "\n")
        f.write("=== CLEANED LLM OUTPUT ===\n")
        f.write(cleaned + "\n")
        f.write("======================\n")

    return {"extracted_json": cleaned, **state}


# Auto-fix common JSON formatting issues from LLM responses.
def auto_fix_json(raw_json: str) -> str:
    """
    Auto-fix common JSON formatting issues from LLM responses.
    
    Args:
        raw_json: Raw JSON string from LLM
        
    Returns:
        Fixed JSON string
    """
    # Step 1: Remove markdown code blocks
    cleaned = raw_json.strip()
    if cleaned.startswith("```json"):
        cleaned = re.sub(r'^```json\s*', '', cleaned)
    elif cleaned.startswith("```"):
        cleaned = re.sub(r'^```\s*', '', cleaned)
    if cleaned.endswith("```"):
        cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Step 2: Fix missing commas after string/object values
    # Pattern: "value"\n"key": -> "value",\n"key":
    cleaned = re.sub(r'(".*?")\s*\n\s*(".*?":\s*)', r'\1,\n\2', cleaned)
    
    # Step 3: Fix missing commas after object closing braces
    # Pattern: }\n"key": -> },\n"key":
    cleaned = re.sub(r'(\})\s*\n\s*(".*?":\s*)', r'\1,\n\2', cleaned)
    
    # Step 4: Fix array syntax - convert numbered object keys to proper arrays
    # Pattern: "tasks":[\n0:{ -> "tasks":[{
    cleaned = re.sub(r'(\[\s*\n\s*)(\d+):\s*\{', r'\1{', cleaned)
    
    # Step 5: Fix numbered object separators in arrays
    # Pattern: }\n1:{ -> },{
    cleaned = re.sub(r'(\}\s*\n\s*)(\d+):\s*\{', r'\1,{', cleaned)
    
    # Step 6: Fix missing commas after arrays
    # Pattern: ]\n"key": -> ],\n"key":
    cleaned = re.sub(r'(\])\s*\n\s*(".*?":\s*)', r'\1,\n\2', cleaned)
    
    # Step 7: Fix email_index with unknown values
    cleaned = re.sub(r'"<unknown>"', r'"unknown"', cleaned)
    
    # Step 8: Remove any remaining numbered labels before array closing
    cleaned = re.sub(r'(\})\s*\n\s*(\d+):\s*(\])', r'\1\2', cleaned)
    
    return cleaned
# Node: write to graph
    # Node: write to graph
def write_graph_node(state):
    print("=== WRITE_GRAPH_NODE STARTED ===")
    extracted = state["validated_json"]
    print(f"Extracted task for Neo4j: {extracted}")
    
    # Ensure email_source is never null
    email_source = extracted.get("Email Source", "")
    if not email_source or email_source == "Unknown":
        email_source = f"extracted-{int(time.time())}"  # Generate a unique ID
    
    neo4j_task = {
        "name": extracted.get("Name", "Unnamed Task"),
        "description": extracted.get("Task Description", ""),
        "due_date": extracted.get("Due Date"),
        "received_date": extracted.get("Received Date"),
        "status": extracted.get("Status"),
        "topic": extracted.get("Topic"),
        "priority": extracted.get("Priority Level"),
        "sender": extracted.get("Sender"),
        "assignee": extracted.get("Assigned To"),
        "email_source": email_source
    }
    print(f"Neo4j task prepared: {neo4j_task}")
    
    try:
        write_tasks_to_neo4j([neo4j_task])
        print(f"✅ Successfully wrote task to Neo4j: {neo4j_task.get('name', 'Unknown')}")
        return {"neo4j_status": "success", "graph": None, **state}
    except Exception as e:
        print(f"❌ Failed to write to Neo4j: {e}")
        return {"neo4j_status": "error", "error": str(e), "graph": None, **state}


# Node: query graph
def query_graph_node(state):
    """Use improved GraphRAG for flexible semantic queries."""
    try:
        # from utils.graphrag import GraphRAG, format_graphrag_response
        
        # Use GraphRAG for semantic querying
        error_msg = "GraphRAG has been removed. Semantic graph queries are not available."
        return {"observation": error_msg, **state}
    except Exception:
        # Simple fallback if GraphRAG fails
        error_msg = "No graph data available. Please ensure graph is loaded."
        return {"observation": error_msg, **state}


# Node: answer generation using graph observation
def answer_node(state):
    date = state.get("current_date", datetime.now().strftime("%Y-%m-%d"))
    user_name = state.get("name", "User")  # Default to "User" if no name
    filled = reason_prompt.format(
        input=state["input"],
        observation=state["observation"],
        current_date=date,
        name=user_name
    )
    response = get_llm().invoke(filled)
    return {"final_answer": response.content, **state}


# Node: conversational LLM with persistent memory
def conversational_llm_node(state):
    """
    LangGraph node for conversational LLM with persistent memory.
    Expects 'conversation_id' and 'user_message' in state.
    """
    conversation_id = state["conversation_id"]
    user_message = state["user_message"]

    # 1. Fetch conversation history from FastAPI
    resp = requests.get(f"http://localhost:8000/chat_turns/{conversation_id}")
    history = resp.json() if resp.ok else []

    # 2. Build prompt with history
    prompt = "Conversation so far:\n"
    for turn in history:
        prompt += f"User: {turn['user_message']}\n"
        prompt += f"Assistant: {turn['assistant_message']}\n"
    prompt += f"User: {user_message}\nAssistant:"

    # 3. Call your LLM (replace with your actual call)
    llm = get_llm()
    result = llm.invoke(prompt)
    assistant_message = result.content.strip()

    # 4. Store the new turn in FastAPI
    requests.post("http://localhost:8000/chat_turns/", json={
        "conversation_id": conversation_id,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "state": state  # Optionally store the full state
    })

    # 5. Return updated state
    return {
        "assistant_message": assistant_message,
        "conversation_id": conversation_id,
        "history": history + [{"user_message": user_message, "assistant_message": assistant_message}],
        **state
    }
