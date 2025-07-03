import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from utils.prompt_template import rag_extraction_prompt, reason_prompt
from utils.embedding import retrieve_similar_chunks
from utils.graph_writer import write_tasks_to_graph
from utils.tools import query_graph, infer_topic_name

load_dotenv()
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

# Node: validate output
def validate_json_node(state):
    import json
    import re
    retry = state.get("retry_count", 0)

    try:
        cleaned_json = state["extracted_json"]

        # Strip code fences from LLM output
        if cleaned_json.startswith("```json"):
            cleaned_json = re.sub(r'^```json\s*', '', cleaned_json)
        if cleaned_json.startswith("```"):
            cleaned_json = re.sub(r'^```\s*', '', cleaned_json)
        if cleaned_json.endswith("```"):
            cleaned_json = re.sub(r'\s*```$', '', cleaned_json)

        parsed = json.loads(cleaned_json)
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
            "name": extracted.get("topic", "Unknown Topic"),
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
                    "email_index": None
                }
            ] if extracted.get("deliverable") else []
        }
    }
    G = write_tasks_to_graph([transformed_data])
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
    filled = reason_prompt.format(input=state["input"], observation=state["observation"], current_date=date)
    response = get_llm().invoke(filled)
    return {"final_answer": response.content, **state}