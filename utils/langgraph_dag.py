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

extraction_workflow = StateGraph(state_schema=ExtractionState)
extraction_workflow.add_node("build_prompt", rag_prompt_node)
extraction_workflow.add_node("extract_json", extract_json_node)

# Wrap validate_json_node to add logging
def validate_json_node_with_logging(state):
    print("VALIDATE:", {
        "retry_count": state.get("retry_count"),
        "extracted_excerpt": state.get("extracted_json", "")[:200]
    })
    return validate_json_node(state)

extraction_workflow.add_node("validate", validate_json_node_with_logging)
extraction_workflow.add_node("write_graph", write_graph_node)

def fail_node(state):
    return {"status": "extraction_failed", **state}

extraction_workflow.add_node("fail", fail_node)

extraction_workflow.set_entry_point("build_prompt")
extraction_workflow.add_edge("build_prompt", "extract_json")
extraction_workflow.add_edge("extract_json", "validate")
def router(state):
    if state.get("valid", False):
        return "write_graph"
    if state.get("retry_count", 0) >= 3:
        return "fail"
    return "extract_json"

extraction_workflow.add_conditional_edges("validate", router, {
    "write_graph": "write_graph",
    "extract_json": "extract_json",
    "fail": "fail"  # gracefully end the graph
})
extraction_workflow.set_finish_point(["write_graph", "fail"])

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
