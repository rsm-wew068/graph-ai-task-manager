import os
from datetime import datetime
from langchain_openai import ChatOpenAI

# Handle dotenv import gracefully for deployment environments
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set directly
    pass

from utils.prompt_template import rag_extraction_prompt, reason_prompt
from utils.embedding import retrieve_similar_chunks
from utils.graph_writer import write_tasks_to_graph
import re
import json

openai_key = os.getenv("OPENAI_API_KEY")


def get_llm():
    return ChatOpenAI(model="gpt-4", temperature=0.2,
                      openai_api_key=openai_key)


# Node: chunked prompt creation for RAG using full email metadata
def rag_prompt_node(state):
    # Get the current email row data
    current_email = state.get("current_email_row", {})
    
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

    return {"extracted_json": raw, **state}


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


# Node: validate output
def validate_json_node(state):
    retry = state.get("retry_count", 0)

    try:
        cleaned_json = state["extracted_json"]

        # Auto-fix common JSON issues
        cleaned_json = auto_fix_json(cleaned_json)
        
        parsed = json.loads(cleaned_json)
        
        # Check if JSON is meaningful (not empty or missing critical fields)
        is_meaningful = False
        
        if "Topic" in parsed:
            # Nested structure - check if it has tasks
            topic = parsed.get("Topic", {})
            tasks = topic.get("tasks", []) if isinstance(topic, dict) else []
            is_meaningful = len(tasks) > 0
        else:
            # Flat structure - check if it has deliverable/topic
            has_deliverable = bool(parsed.get("deliverable", "").strip())
            has_topic = bool(parsed.get("topic", "").strip())
            is_meaningful = has_deliverable or has_topic
        
        if not is_meaningful:
            print("⚠️ JSON extraction contains no meaningful task data")
            # Treat empty/meaningless JSON as failed extraction
            fallback_json = {
                "topic": "Email Review Required",
                "summary": "Empty extraction - needs human review",
                "deliverable": "Manual Review Required",
                "owner": "Unknown",
                "role": "Unknown",
                "department": "Unknown",
                "organization": "Unknown",
                "start_date": None,
                "due_date": None
            }
            return {
                "validated_json": fallback_json,
                "valid": False,  # Trigger HITL
                "needs_human_review": True,
                "retry_count": retry + 1,
                **state
            }
        
        print(f"✅ JSON extraction successful: {parsed}")
        return {
            "validated_json": parsed,
            "valid": True,
            "retry_count": retry,
            **state
        }
    except Exception as e:
        print(f"❌ JSON parsing failed (retry {retry + 1}): {e}")
        print(f"Raw content: {state.get('extracted_json', 'None')[:100]}...")
        
        # Check if error is related to email_index (auto-fix these)
        error_msg = str(e).lower()
        email_keywords = ['email_index', 'index', 'email']
        if any(keyword in error_msg for keyword in email_keywords):
            print("🔧 Auto-fixing email_index related JSON issue...")
            fallback_json = {
                "topic": "Email Review",
                "summary": "Auto-fixed email index issue",
                "deliverable": "Extracted Task",
                "owner": "Unknown",
                "role": "Unknown",
                "department": "Unknown",
                "organization": "Unknown",
                "start_date": None,
                "due_date": None
            }
            return {
                "validated_json": fallback_json,
                "valid": False,  # Changed: Trigger HITL for consistency
                "needs_human_review": True,  # Flag for HITL workflow
                "retry_count": retry + 1,
                **state
            }
        
        if retry >= 2:  # Allow 3 attempts total (0, 1, 2)
            # Failed extractions should trigger human-in-the-loop
            fallback_json = {
                "topic": "Email Review",
                "summary": "Failed to extract structured data from email",
                "deliverable": "Manual Review Required",
                "owner": "Unknown",
                "role": "Unknown",
                "department": "Unknown",
                "organization": "Unknown",
                "start_date": None,
                "due_date": None
            }
            msg = (f"🔄 Using fallback JSON structure after {retry + 1} "
                   f"attempts - triggering HITL")
            print(msg)
            return {
                "validated_json": fallback_json,
                "valid": False,  # Trigger human-in-the-loop
                "needs_human_review": True,  # Flag for HITL workflow
                "retry_count": retry + 1,
                **state
            }
        return {
            "valid": False,
            "retry_count": retry + 1,
            **state
        }


# Node: write to graph
def write_graph_node(state):
    extracted = state["validated_json"]
    
    # Ensure the JSON has the expected nested structure
    if "Topic" not in extracted:
        # Convert flat structure to nested structure
        owner_name = extracted.get("owner", "Unknown Owner")
        owner_role = extracted.get("role", "Unknown Role")
        owner_dept = extracted.get("department", "Unknown Department")
        owner_org = extracted.get("organization", "Unknown Organization")
        
        topic_name = extracted.get("topic", "Unknown Topic")
        
        # Handle collaborators in flat structure
        collaborators = extracted.get("collaborators", [])
        if not isinstance(collaborators, list):
            collaborators = []
        
        transformed_data = {
            "Topic": {
                "name": topic_name,
                "tasks": [
                    {
                        "task": {
                            "name": extracted.get("deliverable",
                                                  "Unnamed Task"),
                            "start_date": extracted.get("start_date"),
                            "due_date": extracted.get("due_date"),
                            "summary": extracted.get("summary", ""),
                            "owner": {
                                "name": owner_name,
                                "role": owner_role,
                                "department": owner_dept,
                                "organization": owner_org
                            },
                            "collaborators": collaborators
                        },
                        "email_index": state.get("email_index")
                    }
                ] if extracted.get("deliverable") else []
            }
        }
    else:
        # JSON already has the correct nested structure
        transformed_data = extracted
    
    G = write_tasks_to_graph(
        [transformed_data],
        save_path="topic_graph.gpickle"
    )
    return {"graph": G, **state}


# Node: query graph
def query_graph_node(state):
    """Use improved GraphRAG for flexible semantic queries."""
    try:
        from utils.graphrag import GraphRAG, format_graphrag_response
        
        # Use GraphRAG for semantic querying
        rag = GraphRAG()
        result = rag.query_with_semantic_reasoning(state["input"])
        
        # Format the response for the LLM
        formatted_response = format_graphrag_response(result)
        
        return {"observation": formatted_response, **state}
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
