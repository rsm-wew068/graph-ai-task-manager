from langgraph.graph import StateGraph
from typing import TypedDict
from datetime import datetime
from utils.langgraph_nodes import (
    topic_inference_node,
    query_graph_node,
    answer_node,
    rag_prompt_node,
    extract_json_node,
    validate_json_node,
    write_graph_node,
)

# Define pipeline state schema
class AppState(TypedDict, total=False):
    input: str
    name: str
    topic_name: str
    current_date: str
    observation: str
    final_answer: str


# Build the workflow
workflow = StateGraph(state_schema=AppState)
workflow.add_node("topic_inference", topic_inference_node)
workflow.add_node("query_graph", query_graph_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("topic_inference")
workflow.add_edge("topic_inference", "query_graph")
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
    email_text: str
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

extraction_workflow = StateGraph(state_schema=ExtractionState)
extraction_workflow.add_node("build_prompt", rag_prompt_node)
extraction_workflow.add_node("extract_json", extract_json_node)

# Split validation into attempt and pause
def attempt_json_parse_node(state):
    """Attempt to parse JSON but don't retry - just mark for review if invalid."""
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
        
        # Basic validation - check if it has expected structure
        if isinstance(parsed, dict) and (
            "name" in parsed or "deliverable" in parsed or "task" in parsed
        ):
            return {
                "validated_json": parsed,
                "valid": True,
                "needs_user_review": False,
                "status": "valid_json",
                **state
            }
        else:
            return {
                "valid": False,
                "needs_user_review": True,
                "status": "invalid_structure",
                **state
            }
            
    except json.JSONDecodeError:
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

# The pause_for_review node will finish the pipeline and return control to UI
extraction_workflow.set_finish_point([
    "write_graph", "fail", "pause_for_review"
])

extraction_app = extraction_workflow.compile()


def run_extraction_pipeline(email_text, faiss_index, all_chunks, email_index):
    state = {
        "email_text": email_text,
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
