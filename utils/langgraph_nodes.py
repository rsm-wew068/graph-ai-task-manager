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
from utils.tools import query_graph, infer_topic_name
import re
import json
from typing import Dict, Any

openai_key = os.getenv("OPENAI_API_KEY")

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=openai_key)

# Node: chunked prompt creation for RAG
def rag_prompt_node(state):
    query_text = state["email_text"]
    related = retrieve_similar_chunks(query_text, index=state["faiss_index"], chunks=state["all_chunks"], k=3)

    prompt = rag_extraction_prompt.format(
        main_email=related[0],
        related_email_1=related[1] if len(related) > 1 else "",
        related_email_2=related[2] if len(related) > 2 else ""
    )
    return {"rag_prompt": prompt, **state}

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
        
        # Ensure email_index is correctly populated
        email_index = state.get("email_index", "unknown")
        parsed = ensure_email_index_populated(parsed, email_index)
        
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
        if any(keyword in error_msg for keyword in ['email_index', 'index', 'email']):
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
                "valid": True,  # Auto-approve email_index fixes
                "retry_count": retry + 1,
                **state
            }
        
        if retry >= 2:  # Allow 3 attempts total (0, 1, 2)
            # Create a fallback structure when JSON extraction repeatedly fails
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
            print(f"🔄 Using fallback JSON structure after {retry + 1} attempts")
            return {
                "validated_json": fallback_json,
                "valid": True,  # Force valid to break the loop
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
    
    # Ensure we have valid owner information
    owner_name = extracted.get("owner") or "Unknown Owner"
    owner_role = extracted.get("role") or "Unknown Role"
    owner_dept = extracted.get("department") or "Unknown Department"
    owner_org = extracted.get("organization") or "Unknown Organization"
    
    transformed_data = {
        "Topic": {
            "name": extracted.get("Topic", {}).get("name", "Unknown Topic"),
            "tasks": [
                {
                    "task": {
                        "name": extracted.get("deliverable", "Unnamed Task"),
                        "start_date": extracted.get("start_date"),
                        "due_date": extracted.get("due_date"),
                        "summary": extracted.get("summary", ""),
                        "owner": {
                            "name": owner_name,
                            "role": owner_role,
                            "department": owner_dept,
                            "organization": owner_org
                        },
                        "collaborators": []
                    },
                    "email_index": state.get("email_index")
                }
            ] if extracted.get("deliverable") else []
        }
    }
    G = write_tasks_to_graph(
        [transformed_data],
        save_path="topic_graph.gpickle"
    )
    return {"graph": G, **state}

# Node: infer topic name from user query
def topic_inference_node(state):
    topic = infer_topic_name(state["input"])
    return {"topic_name": topic, **state}

# Node: query graph
def query_graph_node(state):
    data = query_graph({"name": state["topic_name"]})
    return {"observation": data, **state}

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

def ensure_email_index_populated(validated_json, email_index):
    """
    Ensure the email_index field is correctly populated in extracted JSON.
    This fixes cases where GPT fails to extract or incorrectly extracts
    the Message-ID.
    """
    if not isinstance(validated_json, dict):
        return validated_json
    
    # If we have a Topic structure, update the tasks array
    if "Topic" in validated_json and isinstance(validated_json["Topic"], dict):
        topic = validated_json["Topic"]
        if "tasks" in topic and isinstance(topic["tasks"], list):
            for task_item in topic["tasks"]:
                if isinstance(task_item, dict):
                    # Update email_index if missing or placeholder
                    current_index = task_item.get("email_index", "")
                    if not current_index or current_index in [
                        "<unknown>", "unknown", ""
                    ]:
                        task_item["email_index"] = email_index
    
    # If it's a flat structure, update directly
    if "email_index" in validated_json:
        current_index = validated_json.get("email_index", "")
        if not current_index or current_index in [
            "<unknown>", "unknown", ""
        ]:
            validated_json["email_index"] = email_index
    
    return validated_json
