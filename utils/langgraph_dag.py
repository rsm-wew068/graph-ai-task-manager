from langgraph.graph import StateGraph
from typing import TypedDict
from datetime import datetime
from utils.langgraph_nodes import (
    classify_node_type,
    match_node,
    expand_graph,
    generate_graph_html,
    extract_contexts,
    format_response_node,
    evaluate_ragas,
    return_to_ui,
    # legacy nodes
    advanced_graph_query_node,
    answer_node,
    rag_prompt_node,
    extract_json_node,
    write_graph_node,
)
import json


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
    last_topic: str  # For conversational memory
    last_task: str   # For conversational memory
    conversation_history: list  # For full multi-turn memory


# Build the simplified workflow using GraphRAG
workflow = StateGraph(state_schema=AppState)
workflow.add_node("query_graph", advanced_graph_query_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("query_graph")
workflow.add_edge("query_graph", "answer")
workflow.set_finish_point("answer")

graph_app = workflow.compile()


def run_agent_chat_round(user_query: str) -> dict:
    initial_state = {
        "input": user_query,
        "current_date": datetime.now().strftime("%Y-%m-%d")
    }
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
    try:
        extracted = state.get("extracted_json", "")
        if not extracted:
            return {
                "valid": False,
                "needs_user_review": True,
                "status": "no_json_extracted",
                **state
            }
        import json
        parsed = json.loads(extracted)
        
        # Validate flat structure
        required_fields = ["task_name", "task_description", "topic", "message_id"]
        
        # Check required fields
        for field in required_fields:
            if field not in parsed or not parsed[field]:
                return {"validated_json": {}, "valid": False, "needs_user_review": True, "status": f"Missing required field: {field}", **state}
        
        # Validate data types
        if not isinstance(parsed["task_name"], str):
            return {"validated_json": {}, "valid": False, "needs_user_review": True, "status": "task_name must be a string", **state}
        if not isinstance(parsed["topic"], str):
            return {"validated_json": {}, "valid": False, "needs_user_review": True, "status": "topic must be a string", **state}
        if not isinstance(parsed["message_id"], str):
            return {"validated_json": {}, "valid": False, "needs_user_review": True, "status": "message_id must be a string", **state}
        
        # Set default values for optional fields if not present
        if "status" not in parsed:
            parsed["status"] = "Pending"
        if "priority_level" not in parsed:
            parsed["priority_level"] = "Medium"
        if "spam" not in parsed:
            parsed["spam"] = False
        
        return {
            "validated_json": parsed,
            "valid": True,
            "needs_user_review": False,
            "status": "valid_json",
            **state
        }
    except Exception as e:
        return {"validated_json": {}, "valid": False, "needs_user_review": True, "status": str(e), **state}


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
                "task_name": "Extracted Task",
                "task_description": state.get("extracted_json", "No content")[:100],
                "topic": "General",
                "message_id": state.get("email_index", "unknown"),
                "status": "Pending",
                "priority_level": "Medium",
                "spam": False
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
            "task_name": "Fallback Task",
            "task_description": "Manual correction failed",
            "topic": "General",
            "message_id": state.get("email_index", "unknown"),
            "status": "not started",
            "priority_level": "Medium",
            "spam": False
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
    try:
        # Use the workflow engine to run the full pipeline, including write_graph_node
        result = extraction_app.invoke(state)
        return result
    except Exception as e:
        print(f"[DEBUG] Exception in run_extraction_pipeline: {e}")
        return {"validated_json": {}, "valid": False, "error": str(e)}


def run_extraction_only_pipeline(email_row, faiss_index, all_chunks, email_index):
    """
    Run extraction pipeline WITHOUT storing to databases - only extract and validate.
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
    try:
        # Create a workflow that stops before write_graph_node
        extraction_only_workflow = StateGraph(ExtractionState)
        
        # Add nodes (same as before but without write_graph)
        extraction_only_workflow.add_node("build_prompt", rag_prompt_node)
        extraction_only_workflow.add_node("extract_json", extract_json_node)
        extraction_only_workflow.add_node("attempt_parse", attempt_json_parse_node)
        extraction_only_workflow.add_node("pause_for_review", pause_for_user_review_node)
        extraction_only_workflow.add_node("process_correction", process_user_correction_node)
        
        # Add fail and finish nodes
        def fail_node(state):
            return {"status": "extraction_failed", **state}
        extraction_only_workflow.add_node("fail", fail_node)
        
        def finish_node(state):
            return {"status": "extraction_complete", **state}
        extraction_only_workflow.add_node("finish", finish_node)
        
        def router_extraction_only(state):
            """Route based on validation results - no storage."""
            if state.get("valid", False):
                return "finish"
            elif state.get("needs_user_review", False):
                return "pause_for_review"
            else:
                return "fail"
        
        extraction_only_workflow.add_conditional_edges("attempt_parse", router_extraction_only, {
            "finish": "finish",
            "pause_for_review": "pause_for_review",
            "fail": "fail"
        })
        
        extraction_only_workflow.add_edge("build_prompt", "extract_json")
        extraction_only_workflow.add_edge("extract_json", "attempt_parse")
        extraction_only_workflow.add_edge("pause_for_review", "process_correction")
        extraction_only_workflow.add_edge("process_correction", "attempt_parse")
        
        extraction_only_workflow.set_entry_point("build_prompt")
        extraction_only_workflow.set_finish_point("finish")
        extraction_only_workflow.set_finish_point("fail")
        extraction_only_workflow.set_finish_point("pause_for_review")
        
        extraction_only_app = extraction_only_workflow.compile()
        
        # Run the extraction-only workflow
        result = extraction_only_app.invoke(state)
        return result
    except Exception as e:
        print(f"[DEBUG] Exception in run_extraction_only_pipeline: {e}")
        return {"validated_json": {}, "valid": False, "error": str(e)}


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

# --- New: Fully Graph-Native Entity-Agnostic QA Pipeline ---
from langgraph.graph import StateGraph

class QAState(TypedDict, total=False):
    input: str
    entity_types: list
    entity_values: list
    matched_node_ids: list
    matched_names: list
    all_nodes: list
    graph_html: str
    contexts: list
    final_answer: str
    ragas_scores: dict
    ragas_summary: str
    # Add more as needed

question_workflow = StateGraph(state_schema=QAState)
question_workflow.add_node("classify_node_type", classify_node_type)
question_workflow.add_node("match_node", match_node)
question_workflow.add_node("expand_graph", expand_graph)
question_workflow.add_node("generate_graph_html", generate_graph_html)
question_workflow.add_node("extract_contexts", extract_contexts)
question_workflow.add_node("format_response_node", format_response_node)
question_workflow.add_node("evaluate_ragas", evaluate_ragas)
question_workflow.add_node("return_to_ui", return_to_ui)

question_workflow.set_entry_point("classify_node_type")
question_workflow.add_edge("classify_node_type", "match_node")
question_workflow.add_edge("match_node", "expand_graph")
question_workflow.add_edge("expand_graph", "generate_graph_html")
question_workflow.add_edge("generate_graph_html", "extract_contexts")
question_workflow.add_edge("extract_contexts", "format_response_node")
question_workflow.add_edge("format_response_node", "evaluate_ragas")
question_workflow.add_edge("evaluate_ragas", "return_to_ui")
question_workflow.set_finish_point("return_to_ui")

question_answering_app = question_workflow.compile()

def run_question_answering_pipeline(user_query: str) -> dict:
    initial_state = {"input": user_query}
    return question_answering_app.invoke(initial_state)
