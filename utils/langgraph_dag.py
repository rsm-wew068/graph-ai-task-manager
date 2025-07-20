from langgraph.graph import StateGraph
from typing import TypedDict
from datetime import datetime
from utils.langgraph_nodes import (
    query_graph_node,
    answer_node,
    rag_prompt_node,
    extract_json_node,
    write_graph_node,
    write_notion_node,
)
import requests


def is_meaningful_task(task_name: str, task_obj: dict) -> bool:
    """
    Simplified validation - only reject truly empty or generic fallback tasks.
    
    Args:
        task_name: The name/title of the task
        task_obj: The full task object with additional context
        
    Returns:
        True if task seems meaningful, False if it should trigger HITL
    """
    if not task_name or len(task_name.strip()) <= 3:
        return False
    
    # Only reject obvious fallback/error cases
    fallback_names = [
        "unnamed task", "extracted task", "manual review required",
        "email review", "no content", "unknown task"
    ]
    
    name_lower = task_name.lower().strip()
    
    # Only reject exact matches to fallback names
    if name_lower in fallback_names:
        return False
    
    # Accept everything else - let humans decide during HITL if needed
    return True


# Define pipeline state schema
class AppState(TypedDict, total=False):
    input: str
    name: str
    topic_name: str
    current_date: str
    observation: str
    final_answer: str


# Build the simplified workflow using GraphRAG
workflow = StateGraph(state_schema=AppState)
workflow.add_node("query_graph", query_graph_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("query_graph")
workflow.add_edge("query_graph", "answer")
workflow.set_finish_point("answer")

graph_app = workflow.compile()


def run_agent_chat_round(user_query: str, conversation_id: str = None) -> dict:
    initial_state = {
        "input": user_query,
        "current_date": datetime.now().strftime("%Y-%m-%d")
    }
    # If conversation_id is provided, fetch history from FastAPI and add to state
    if conversation_id:
        API_URL = "http://localhost:8000"  # Change if needed
        resp = requests.get(f"{API_URL}/chat_turns/{conversation_id}")
        history = resp.json() if resp.ok else []
        initial_state["conversation_id"] = conversation_id
        initial_state["history"] = history
    return graph_app.invoke(initial_state)


# Extraction pipeline schema
class ExtractionState(TypedDict, total=False):
    current_email_row: dict  # Full email row with all metadata
    email_text: str  # Backward compatibility - content only
    faiss_index: object
    all_chunks: list
    rag_prompt: str
    extracted_json: str
    validated_json: dict
    valid: bool
    graph: object
    retry_count: int
    needs_user_review: bool
    user_corrected_json: str
    status: str
    email_index: str  # Message-ID for tracking


extraction_workflow = StateGraph(state_schema=ExtractionState)
extraction_workflow.add_node("build_prompt", rag_prompt_node)
extraction_workflow.add_node("extract_json", extract_json_node)

# Split validation into attempt and pause


def normalize_data_types(parsed: dict) -> dict:
    """
    Normalize data types to ensure consistency across the entire pipeline.
    Converts all values to the expected types for Notion database compatibility.
    """
    # Create a copy to avoid modifying the original
    normalized = dict(parsed)
    
    # Define expected types for each field based on Notion schema
    field_types = {
        "Name": str,                    # title
        "Task Description": str,        # rich_text
        "Due Date": str,                # date
        "Received Date": str,           # date
        "Status": str,                  # status (Not started, In progress, Done)
        "Topic": str,                   # select
        "Priority Level": str,          # select
        "Sender": str,                  # email
        "Assigned To": str,             # email
        "Email Source": str,            # url
        "Spam": bool                    # checkbox
    }
    
    # Convert each field to the expected type
    for field_name, expected_type in field_types.items():
        if field_name in normalized:
            value = normalized[field_name]
            
            # Handle None values
            if value is None:
                if expected_type == bool:
                    normalized[field_name] = False
                else:
                    normalized[field_name] = ""
                continue
            
            # Convert to expected type
            if expected_type == str:
                # Convert various types to string
                if isinstance(value, (int, float)):
                    normalized[field_name] = str(value)
                elif isinstance(value, bool):
                    normalized[field_name] = str(value).lower()
                elif isinstance(value, (list, dict)):
                    # For complex types, convert to JSON string
                    import json
                    try:
                        normalized[field_name] = json.dumps(value)
                    except:
                        normalized[field_name] = str(value)
                elif isinstance(value, str):
                    # Clean up string values
                    cleaned = value.strip()
                    if cleaned.lower() in ['null', 'none', 'undefined', '']:
                        normalized[field_name] = ""
                    else:
                        normalized[field_name] = cleaned
                else:
                    # For any other type, convert to string
                    normalized[field_name] = str(value)
            
            elif expected_type == bool:
                # Convert to boolean
                if isinstance(value, bool):
                    normalized[field_name] = value
                elif isinstance(value, str):
                    # Convert string to boolean
                    cleaned = value.strip().lower()
                    if cleaned in ['true', '1', 'yes', 'on']:
                        normalized[field_name] = True
                    elif cleaned in ['false', '0', 'no', 'off', '']:
                        normalized[field_name] = False
                    else:
                        # Default to False for unknown values
                        normalized[field_name] = False
                elif isinstance(value, (int, float)):
                    # Convert number to boolean
                    normalized[field_name] = bool(value)
                else:
                    # Default to False for other types
                    normalized[field_name] = False
    
    return normalized


def normalize_field_names(parsed: dict) -> dict:
    """
    Normalize field names to ensure consistency across the entire pipeline.
    Maps various possible field names to the standard Notion database field names.
    """
    # Define field name mappings (alternative_name -> standard_name)
    # Based on exact Notion database schema
    field_mappings = {
        # Email Source variations
        "email_index": "Email Source",
        "email_source": "Email Source",
        "message_id": "Email Source",
        "email_id": "Email Source",
        
        # Task Name variations
        "task_name": "Name",
        "title": "Name",
        "task_title": "Name",
        
        # Task Description variations
        "task_description": "Task Description",
        "description": "Task Description",
        "summary": "Task Description",
        "content": "Task Description",
        
        # Due Date variations
        "due_date": "Due Date",
        "deadline": "Due Date",
        "due": "Due Date",
        
        # Received Date variations
        "received_date": "Received Date",
        "date_received": "Received Date",
        "email_date": "Received Date",
        
        # Status variations
        "task_status": "Status",
        "state": "Status",
        
        # Topic variations
        "category": "Topic",
        "project": "Topic",
        "subject": "Topic",
        
        # Priority Level variations
        "priority_level": "Priority Level",
        "priority": "Priority Level",
        "urgency": "Priority Level",
        
        # Sender variations
        "sender_email": "Sender",
        "from": "Sender",
        "from_email": "Sender",
        
        # Assigned To variations
        "assigned_to": "Assigned To",
        "assignee": "Assigned To",
        "assignee_email": "Assigned To",
        "to": "Assigned To",
        "responsible": "Assigned To",
        
        # Spam variations
        "is_spam": "Spam",
        "spam_flag": "Spam",
        "spam_detected": "Spam"
    }
    
    # Create a copy to avoid modifying the original
    normalized = dict(parsed)
    
    # Apply field name mappings
    for alt_name, std_name in field_mappings.items():
        if alt_name in normalized and std_name not in normalized:
            normalized[std_name] = normalized.pop(alt_name)
    
    return normalized


def attempt_json_parse_node(state):
    """
    TEMPORARY: Accept any parsed JSON as valid, skipping all validation.
    """
    print("=== ATTEMPT_JSON_PARSE_NODE STARTED ===")
    import json
    extracted = state.get("extracted_json", "")
    print(f"Extracted JSON length: {len(extracted)}")
    print(f"Extracted JSON preview: {extracted[:200]}...")
    
    if not extracted:
        print("No extracted JSON found")
        return {
            "valid": False,
            "needs_user_review": True,
            "status": "no_json_extracted",
            **state
        }
    try:
        print("Parsing JSON...")
        parsed = json.loads(extracted)
        print(f"JSON parsed successfully. Keys: {list(parsed.keys())}")
        print("VALIDATION BYPASSED: Accepting all parsed JSON as valid.")
        
        # Normalize field names to ensure consistency
        print("Normalizing field names...")
        parsed = normalize_field_names(parsed)
        print(f"After field normalization. Keys: {list(parsed.keys())}")
        
        # Normalize data types to ensure consistency
        print("Normalizing data types...")
        parsed = normalize_data_types(parsed)
        print(f"After data type normalization. Keys: {list(parsed.keys())}")
        
        # Fill missing optional fields with None
        optional_fields = [
            "Due Date", "Received Date", "Status", "Topic", "Priority Level", "Sender", "Assigned To", "Email Source", "Spam"
        ]
        for f in optional_fields:
            if f not in parsed:
                if f == "Spam":
                    parsed[f] = False
                elif f == "Status":
                    parsed[f] = "Not started"
                else:
                    parsed[f] = ""
        print(f"After filling optional fields. Keys: {list(parsed.keys())}")
        
        # Convert any datetime/date objects to strings
        from utils.langgraph_nodes import convert_dates_to_strings
        parsed = convert_dates_to_strings(parsed)
        print(f"After date conversion. Final keys: {list(parsed.keys())}")
        
        result = {
            "validated_json": parsed,
            "valid": True,
            "needs_user_review": False,
            "status": "valid_json",
            **state
        }
        print(f"Returning result with keys: {list(result.keys())}")
        return result
        
    except Exception as e:
        print(f"JSON parse error: {e}")
        print(f"CLEANED extracted_json that failed to parse: {extracted}")
        import traceback
        traceback.print_exc()
        return {
            "valid": False,
            "needs_user_review": True,
            "status": "json_parse_error",
            **state
        }


def pause_for_user_review_node(state):
    """This node represents a pause - returns state for UI handling."""
    new_state = dict(state)  # Create a copy
    new_state.update({
        "status": "awaiting_user_review",
        "needs_user_review": True
    })
    return new_state


def process_user_correction_node(state):
    """Process user-corrected JSON."""
    try:
        corrected_json = state.get("user_corrected_json", "")
        new_state = dict(state)  # Create a copy
        if corrected_json:
            import json
            parsed = json.loads(corrected_json)
            
            # Normalize field names to ensure consistency
            parsed = normalize_field_names(parsed)
            
            # Normalize data types to ensure consistency
            parsed = normalize_data_types(parsed)
            
            new_state.update({
                "validated_json": parsed,
                "valid": True,
                "needs_user_review": False,
                "status": "user_corrected"
            })
            return new_state
        else:
            # User chose to skip - create minimal fallback (matches Notion schema exactly)
            fallback = {
                "Name": "Extracted Task",
                "Task Description": state.get("extracted_json", "No content")[:100],
                "Due Date": "",
                "Received Date": "",
                "Status": "Not started",
                "Topic": "",
                "Priority Level": "",
                "Sender": "",
                "Assigned To": "",
                "Email Source": "",
                "Spam": False
            }
            new_state.update({
                "validated_json": fallback,
                "valid": True,
                "needs_user_review": False,
                "status": "user_skipped"
            })
            return new_state
    except Exception:
        # Still invalid after user correction - use fallback (matches Notion schema exactly)
        fallback = {
            "Name": "Fallback Task",
            "Task Description": "Manual correction failed",
            "Due Date": "",
            "Received Date": "",
            "Status": "Not started",
            "Topic": "",
            "Priority Level": "",
            "Sender": "",
            "Assigned To": "",
            "Email Source": "",
            "Spam": False
        }
        new_state = dict(state)
        new_state.update({
            "validated_json": fallback,
            "valid": True,
            "needs_user_review": False,
            "status": "correction_failed"
        })
        return new_state



extraction_workflow.add_node("attempt_parse", attempt_json_parse_node)
extraction_workflow.add_node("pause_for_review", pause_for_user_review_node)
extraction_workflow.add_node("process_correction", process_user_correction_node)
extraction_workflow.add_node("write_notion", write_notion_node)
extraction_workflow.add_node("write_graph", write_graph_node)


def fail_node(state):
    return {"status": "extraction_failed", **state}


extraction_workflow.add_node("fail", fail_node)

extraction_workflow.set_entry_point("build_prompt")
extraction_workflow.add_edge("build_prompt", "extract_json")
extraction_workflow.add_edge("extract_json", "attempt_parse")



# Route: after validation, write to Notion, then Neo4j
def router(state):
    """Route based on validation results and user review needs."""
    if state.get("valid", False):
        return "write_notion"
    elif state.get("needs_user_review", False):
        return "pause_for_review"
    else:
        return "fail"



extraction_workflow.add_conditional_edges("attempt_parse", router, {
    "write_notion": "write_notion",
    "pause_for_review": "pause_for_review",
    "fail": "fail"
})

# After Notion write, always write to Neo4j
extraction_workflow.add_edge("write_notion", "write_graph")


# Set finish points - different nodes can end the workflow
for endpoint in ["write_graph", "fail", "pause_for_review"]:
    extraction_workflow.set_finish_point(endpoint)

extraction_app = extraction_workflow.compile()


def run_extraction_pipeline(email_row, faiss_index, all_chunks, email_index):
    """
    Run extraction pipeline with full email row metadata.
    
    Args:
        email_row: Dictionary containing all email fields (Message-ID, From,
                  To, Subject, content, etc.)
        faiss_index: FAISS index for similarity search
        all_chunks: List of text chunks for RAG
        email_index: Message-ID or unique identifier for this email
    """
    print(f"=== RUN_EXTRACTION_PIPELINE STARTED ===")
    print(f"Email index: {email_index}")
    print(f"Email row keys: {list(email_row.keys())}")
    print(f"FAISS index type: {type(faiss_index)}")
    print(f"All chunks length: {len(all_chunks)}")
    
    try:
        state = {
            "current_email_row": email_row,
            "email_text": email_row.get("content", ""),  # Backward compatibility
            "faiss_index": faiss_index,
            "all_chunks": all_chunks,
            "email_index": email_index,
            "retry_count": 0
        }
        print(f"State created with keys: {list(state.keys())}")
        
        print("Invoking extraction_app...")
        result = extraction_app.invoke(state)
        print(f"Extraction completed. Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"Result status: {result.get('status', 'No status')}")
        
        return result
        
    except Exception as e:
        print(f"Error in run_extraction_pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a fallback result
        return {
            "status": "error",
            "error": str(e),
            "valid": False,
            "needs_user_review": True,
            "email_index": email_index
        }


def resume_extraction_pipeline_with_correction(
    paused_state, user_corrected_json
):
    """Resume the extraction pipeline with user-corrected JSON."""
    # Update the state with user correction
    updated_state = {
        **paused_state,
        "user_corrected_json": user_corrected_json,
        "needs_user_review": False  # Clear the review flag
    }
    
    # Resume from the process_correction node
    # We need to create a new graph starting from process_correction
    resume_workflow = StateGraph(state_schema=ExtractionState)
    resume_workflow.add_node(
        "process_correction", process_user_correction_node
    )
    resume_workflow.add_node("write_graph", write_graph_node)
    resume_workflow.add_node("fail", fail_node)
    
    resume_workflow.set_entry_point("process_correction")
    resume_workflow.add_edge("process_correction", "write_graph")
    resume_workflow.set_finish_point(["write_graph", "fail"])
    
    resume_app = resume_workflow.compile()
    return resume_app.invoke(updated_state)
