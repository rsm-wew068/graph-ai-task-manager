from langgraph.graph import StateGraph
from typing import TypedDict
from datetime import datetime
from utils.langgraph_nodes import (
    query_graph_node,
    answer_node,
    rag_prompt_node,
    extract_json_node,
    write_graph_node,
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


def attempt_json_parse_node(state):
    """
    Attempt to parse JSON but don't retry - just mark for review if invalid.
    """
    try:
        extracted = state.get("extracted_json", "")
        if not extracted:
            return {
                "valid": False,
                "needs_user_review": True,
                "status": "no_json_extracted",
                **state
            }
        
        # Try to parse the JSON
        import json
        parsed = json.loads(extracted)
        
        # Enhanced validation for meaningful task content
        if isinstance(parsed, dict):
            # Check for nested structure (preferred)
            if "Topic" in parsed and "tasks" in parsed.get("Topic", {}):
                tasks = parsed["Topic"]["tasks"]
                if tasks and isinstance(tasks, list) and len(tasks) > 0:
                    # Check if first task has meaningful content
                    first_task = tasks[0]
                    if isinstance(first_task, dict) and "task" in first_task:
                        task_obj = first_task["task"]
                        task_name = task_obj.get("name", "")
                        
                        # Enhanced validation criteria
                        if not is_meaningful_task(task_name, task_obj):
                            # Don't trigger HITL for unmeaningful tasks, just auto-fix
                            task_obj["name"] = f"Email Task: {task_obj.get('summary', 'Extracted from email')[:50]}"
                        
                        return {
                            "validated_json": parsed,
                            "valid": True,
                            "needs_user_review": False,
                            "status": "valid_json",
                            **state
                        }
            
            # Check for flat structure with meaningful content
            elif ("deliverable" in parsed or "name" in parsed):
                task_name = parsed.get("deliverable") or parsed.get("name", "")
                
                # Enhanced validation for flat structure
                if not is_meaningful_task(task_name, parsed):
                    # Auto-fix instead of triggering HITL
                    parsed["deliverable"] = f"Email Task: {parsed.get('summary', 'Extracted from email')[:50]}"
                
                return {
                    "validated_json": parsed,
                    "valid": True,
                    "needs_user_review": False,
                    "status": "valid_json",
                    **state
                }
        
        # If we get here, JSON structure is unusual but parseable
        # Create a basic fallback instead of triggering HITL
        fallback_json = {
            "name": "Email Review Task",
            "deliverable": "Review and process email content",
            "summary": str(parsed)[:100] + "..." if len(str(parsed)) > 100 else str(parsed),
            "owner": "Unknown",
            "role": "Unknown", 
            "department": "Unknown",
            "organization": "Unknown",
            "start_date": "Unknown",
            "due_date": "Unknown"
        }
        
        return {
            "validated_json": fallback_json,
            "valid": True,
            "needs_user_review": False,
            "status": "auto_fallback_created",
            **state
        }
            
    except json.JSONDecodeError:
        # JSON parsing failed - create an editable template from the original extraction
        raw_json = state.get("extracted_json", "")
        
        # Try to auto-fix common issues for user editing
        if raw_json:
            try:
                # Replace null values with "Unknown" to make it parseable for editing
                fixed_json = raw_json.replace('null', '"Unknown"').replace('NULL', '"Unknown"')
                # Try to parse the fixed version
                import json
                test_parse = json.loads(fixed_json)
                # If successful, provide this as the correctable template
                return {
                    "valid": False,
                    "needs_user_review": True,
                    "status": "json_parse_error",
                    "correctable_json": fixed_json,  # Provide correctable version
                    **state
                }
            except:
                pass
        
        # If auto-fix failed, still provide the original for manual editing
        return {
            "valid": False,
            "needs_user_review": True,
            "status": "json_parse_error",
            "correctable_json": raw_json,  # At least show the original
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
            new_state.update({
                "validated_json": parsed,
                "valid": True,
                "needs_user_review": False,
                "status": "user_corrected"
            })
            return new_state
        else:
            # User chose to skip - create minimal fallback
            fallback = {
                "name": "Extracted Task",
                "deliverable": state.get("extracted_json", "No content")[:100]
            }
            new_state.update({
                "validated_json": fallback,
                "valid": True,
                "needs_user_review": False,
                "status": "user_skipped"
            })
            return new_state
    except Exception:
        # Still invalid after user correction - use fallback
        fallback = {
            "name": "Fallback Task",
            "deliverable": "Manual correction failed"
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
extraction_workflow.add_node(
    "process_correction", process_user_correction_node
)
extraction_workflow.add_node("write_graph", write_graph_node)


def fail_node(state):
    return {"status": "extraction_failed", **state}


extraction_workflow.add_node("fail", fail_node)

extraction_workflow.set_entry_point("build_prompt")
extraction_workflow.add_edge("build_prompt", "extract_json")
extraction_workflow.add_edge("extract_json", "attempt_parse")


def router(state):
    """Route based on validation results and user review needs."""
    if state.get("valid", False):
        return "write_graph"
    elif state.get("needs_user_review", False):
        return "pause_for_review"
    else:
        return "fail"


extraction_workflow.add_conditional_edges("attempt_parse", router, {
    "write_graph": "write_graph",
    "pause_for_review": "pause_for_review",
    "fail": "fail"
})

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
    state = {
        "current_email_row": email_row,
        "email_text": email_row.get("content", ""),  # Backward compatibility
        "faiss_index": faiss_index,
        "all_chunks": all_chunks,
        "email_index": email_index,
        "retry_count": 0
    }
    return extraction_app.invoke(state)


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
