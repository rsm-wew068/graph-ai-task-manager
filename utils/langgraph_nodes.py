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
# Old import removed - using flat structure methods directly
from utils.database import store_validated_tasks
import re
import json
import openai
from neo4j import GraphDatabase
from openai import OpenAI

def validate_cypher(cypher):
    # Only allow read-only queries
    forbidden = ['CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP', 'CALL db.', 'LOAD CSV']
    for word in forbidden:
        if re.search(r'\b' + word + r'\b', cypher, re.IGNORECASE):
            return False
    return True

def fetch_task_summary_from_neo4j(driver, task_name):
    with driver.session() as session:
        result = session.run(
            "MATCH (t:Task {name: $task_name}) RETURN t.summary AS summary",
            task_name=task_name
        )
        record = result.single()
        return record["summary"] if record else None

def advanced_graph_query_node(state):
    user_query = state.get("input", "")
    openai_key = os.getenv("OPENAI_API_KEY")

    # 1. Retrieve context from GraphRAG (ChainQA: open-ended)
    from utils.neo4j_graphrag import Neo4jGraphRAG
    rag = Neo4jGraphRAG()
    # Use flexible retrieval for ChainQA
    graph_result = rag.query_flexible(user_query)
    context = graph_result.get('all_nodes', [])
    formatted_context = '\n'.join(context) if context else 'No relevant graph data found.'

    # 2. Generate answer directly from graph context (skip ChainQA LLM)
    # --- ChainQA LLM call skipped ---
    # from openai import OpenAI
    # client = OpenAI(api_key=openai_key)
    # answer_prompt = f"""
    # You are an expert assistant. Given the user question and the following graph data, generate a clear, concise answer for the user.
    # User question: {user_query}
    # Graph data:
    # {formatted_context}
    # Answer:
    # """
    # answer_response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": answer_prompt}]
    # )
    # final_answer = answer_response.choices[0].message.content.strip()

    # Instead, compose a simple answer from the graph context
    if context and context != ['No relevant graph data found.']:
        final_answer = f"Here is what was found in the graph for your question:\n{formatted_context}"
    else:
        final_answer = "No relevant information was found in the graph for your question."

    # 3. RAGAS evaluation (as before, on the composed answer)
    import threading
    import logging
    logger = logging.getLogger(__name__)
    ragas_scores = None
    ragas_summary = None
    def run_ragas_eval():
        try:
            from utils.ragas_evaluator import RAGASEvaluator
            import asyncio
            evaluator = RAGASEvaluator()
            ragas_scores_local = asyncio.run(
                evaluator.evaluate_single(user_query, final_answer, [formatted_context], ground_truth=formatted_context)
            )
            nonlocal ragas_scores, ragas_summary
            ragas_scores = ragas_scores_local
            if ragas_scores and all(isinstance(v, float) for v in ragas_scores.values()):
                ragas_summary = evaluator.format_evaluation_summary(ragas_scores)
            else:
                ragas_summary = "⚠️ RAGAS evaluation unavailable: No valid scores returned."
        except Exception as e:
            logger.error(f"[RAGAS] Exception: {e}")
            ragas_scores = {}
            ragas_summary = f"⚠️ RAGAS evaluation unavailable: {e}"
    threading.Thread(target=run_ragas_eval, daemon=True).start()

    return {
        "observation": final_answer,
        "graph_context": formatted_context,
        "ragas_scores": ragas_scores if ragas_scores is not None else {},
        "ragas_summary": ragas_summary if ragas_summary is not None else "RAGAS evaluation in progress...",
        **state
    }


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
            "Please set it in your Hugging Face Space settings."
        )
    return ChatOpenAI(model="gpt-4o", temperature=0.2,
                      openai_api_key=openai_key)


# Node: chunked prompt creation for RAG using full email metadata
def rag_prompt_node(state):
    print("[DEBUG] Entering rag_prompt_node. State keys:", list(state.keys()))
    try:
        # Get the current email row data
        current_email = state.get("current_email_row", {})
        print("[DEBUG] current_email in rag_prompt_node:", current_email)
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
            related_email_1 = related[0] if len(related) > 0 else ""
            related_email_2 = related[1] if len(related) > 1 else ""
        else:
            related_email_1 = ""
            related_email_2 = ""
        from utils.prompt_template import example_json
        prompt = rag_extraction_prompt.format(
            main_email=main_email_context,
            related_email_1=related_email_1,
            related_email_2=related_email_2,
            example_json=example_json
        )
        print("[DEBUG] Prompt constructed in rag_prompt_node:", prompt[:500], "... (truncated)")
        result = {"rag_prompt": prompt, **state}
        print("[DEBUG] Exiting rag_prompt_node. Result keys:", list(result.keys()))
        return result
    except Exception as e:
        print(f"[DEBUG] Exception in rag_prompt_node: {e}")
        raise


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
    
    # Extract all available fields - handle both parquet and database column names
    message_id = email_row.get("message_id") or email_row.get("Message-ID", "Unknown")
    date = email_row.get("date_received") or email_row.get("Date", "Unknown")
    from_email = email_row.get("from_email") or email_row.get("From", "Unknown")
    to_email = email_row.get("to_email") or email_row.get("To", "Unknown")
    cc_email = email_row.get("cc_email") or email_row.get("Cc", "None") or "None"
    bcc_email = email_row.get("bcc_email") or email_row.get("Bcc", "None") or "None"
    from_name = email_row.get("from_name") or email_row.get("Name-From", "Unknown")
    to_name = email_row.get("to_name") or email_row.get("Name-To", "Unknown")
    cc_name = email_row.get("cc_name") or email_row.get("Name-Cc", "None") or "None"
    bcc_name = email_row.get("bcc_name") or email_row.get("Name-Bcc", "None") or "None"
    subject = email_row.get("subject") or email_row.get("Subject", "No Subject")
    content = email_row.get("content", "No content")
    
    # Debug print for all fields
    print("[DEBUG] format_email_for_llm fields:", {
        'message_id': message_id, 'date': date, 'from_email': from_email, 'to_email': to_email,
        'cc_email': cc_email, 'bcc_email': bcc_email, 'from_name': from_name, 'to_name': to_name,
        'cc_name': cc_name, 'bcc_name': bcc_name, 'subject': subject, 'content': content
    })
    
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
    print("\n[DEBUG] extract_json_node called. State keys:", list(state.keys()))
    llm = get_llm()
    print("[DEBUG] Prompt sent to LLM:\n", state.get("rag_prompt"))
    result = llm.invoke(state["rag_prompt"])
    print("[DEBUG] LLM result object:", result)
    raw = result.content.strip()
    print("[DEBUG] Raw LLM output from extract_json_node:\n", raw)

    # Auto-strip common GPT wrapping
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    # Always print the raw output for debugging
    print("[DEBUG] (Post-strip) Raw LLM output:\n", raw)

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
        cleaned_json = auto_fix_json(cleaned_json)
        parsed = json.loads(cleaned_json)

        # Validate flat structure
        required_fields = ["task_name", "task_description", "topic", "message_id"]
        optional_fields = ["due_date", "received_date", "status", "priority_level", "sender", "assigned_to", "spam", "validation_status", "confidence_score"]
        
        # Set default values for optional fields if not present
        if "status" not in parsed:
            parsed["status"] = "not started"
        elif parsed["status"] not in ["not started", "in progress", "completed"]:
            parsed["status"] = "not started"  # Default if invalid status
        if "priority_level" not in parsed:
            parsed["priority_level"] = "Medium"
        if "spam" not in parsed:
            parsed["spam"] = False
        
        # Check required fields
        for field in required_fields:
            if field not in parsed or not parsed[field]:
                print(f"[DEBUG] validate_json_node: Missing required field '{field}'. Parsed:", parsed)
                return {"validated_json": {}, "valid": False, "error": f"Missing required field: {field}", **state}
        
        # Validate data types
        if not isinstance(parsed["task_name"], str):
            return {"validated_json": {}, "valid": False, "error": "task_name must be a string", **state}
        if not isinstance(parsed["topic"], str):
            return {"validated_json": {}, "valid": False, "error": "topic must be a string", **state}
        if not isinstance(parsed["message_id"], str):
            return {"validated_json": {}, "valid": False, "error": "message_id must be a string", **state}
        
        return {"validated_json": parsed, "valid": True, **state}
    except Exception as e:
        print(f"[DEBUG] Exception in validate_json_node: {e}")
        return {"validated_json": {}, "valid": False, "error": str(e), **state}


# Node: write to graph and database
def write_graph_node(state):
    extracted = state["validated_json"]
    
    # Store flat task data directly in PostgreSQL database
    database_success = False
    try:
        task_ids = store_validated_tasks([extracted])
        print(f"✅ Stored {len(task_ids)} tasks in PostgreSQL database")
        database_success = len(task_ids) > 0
    except Exception as e:
        print(f"⚠️ Warning: Failed to store in PostgreSQL: {e}")
        # Continue with Neo4j storage even if PostgreSQL fails
    
    # Store in Neo4j graph database using flat structure
    graph_success = False
    try:
        print(">>> About to write to Neo4j:", extracted)  # DEBUG PRINT
        
        # Use the flat structure method for Neo4j
        from utils.neo4j_graph_writer import Neo4jGraphWriter
        neo4j_writer = Neo4jGraphWriter()
        graph_success = neo4j_writer.write_tasks_from_table([extracted], clear_existing=False)
        
        print(">>> Write to Neo4j success:", graph_success)  # DEBUG PRINT
        if graph_success:
            print("✅ Stored tasks in Neo4j graph database")
        else:
            print("⚠️ Warning: Failed to store in Neo4j")
    except Exception as e:
        print(f"⚠️ Warning: Neo4j storage failed: {e}")
        graph_success = False
    
    return {"graph_stored": graph_success, "database_stored": database_success, **state}


# Node: query graph
def query_graph_node(state):
    """Use improved Neo4j GraphRAG for flexible semantic queries."""
    try:
        from utils.neo4j_graphrag import Neo4jGraphRAG, format_graphrag_response
        
        # Use Neo4j GraphRAG for semantic querying
        rag = Neo4jGraphRAG()
        result = rag.query_with_semantic_reasoning(state["input"])
        
        # Format the response for the LLM
        formatted_response = format_graphrag_response(result)
        
        return {"observation": formatted_response, **state}
    except Exception as e:
        # Fallback with more detailed error info
        error_msg = f"No Neo4j graph data available. Error: {str(e)}. Please ensure Neo4j is running and graph is loaded."
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


def normalize_task_dates(task_json):
    if "sent_date" in task_json:
        task_json["sent_date"] = task_json.pop("sent_date")
    # Recursively handle nested structures if needed
    if "Topic" in task_json and "tasks" in task_json["Topic"]:
        for t in task_json["Topic"]["tasks"]:
            if "task" in t and "sent_date" in t["task"]:
                t["task"]["sent_date"] = t["task"].pop("sent_date")
    return task_json

def classify_node_type(state):
    """LLM-based node to classify the user question as about a Topic, Person, Task, etc."""
    import os
    from openai import OpenAI
    question = state.get("input", "")
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are an expert entity classifier for a knowledge graph. "
        "Given a user question, classify what entity type(s) it is about (Person, Task, Topic, Date, Organization, etc.), "
        "and extract the main entity value(s) mentioned. "
        "Return a JSON object with keys: entity_types (list of strings), entity_values (list of strings).\n"
        "\n"
        "Examples:\n"
        "Q: What did Alice assign Bob?\n"
        "A: {\"entity_types\": [\"Person\"], \"entity_values\": [\"Alice\", \"Bob\"]}\n"
        "Q: What tasks are due next week?\n"
        "A: {\"entity_types\": [\"Task\", \"Date\"], \"entity_values\": [\"next week\"]}\n"
        "Q: Who owns the Project Phoenix topic?\n"
        "A: {\"entity_types\": [\"Topic\", \"Person\"], \"entity_values\": [\"Project Phoenix\"]}\n"
        f"Q: {question}\nA: "
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        import json
        try:
            parsed = json.loads(answer)
        except Exception:
            import re
            match = re.search(r'\{.*\}', answer)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = {"entity_types": [], "entity_values": []}
        entity_types = parsed.get("entity_types", [])
        entity_values = parsed.get("entity_values", [])
        print(f"[DEBUG] classify_node_type: entity_types={entity_types}, entity_values={entity_values}")
        return {"entity_types": entity_types, "entity_values": entity_values, **state}
    except Exception as e:
        print(f"[DEBUG] classify_node_type exception: {e}")
        return {"entity_types": [], "entity_values": [], **state}

def match_node(state):
    """Embed query and match to node(s) of the classified type in Neo4j."""
    from utils.neo4j_graphrag import Neo4jGraphRAG
    entity_types = state.get("entity_types", [])
    entity_values = state.get("entity_values", [])
    rag = Neo4jGraphRAG()
    matched_node_ids = []
    matched_names = []
    for entity_value in entity_values:
        # Use flexible query to match any node type
        result = rag.query_flexible(entity_value)
        if result.get("all_nodes"):
            # Use the first matched node as entry point
            if result.get("matched_nodes"):
                matched_names.extend(result["matched_nodes"])
            if result.get("matched_node_ids"):
                matched_node_ids.extend(result["matched_node_ids"])
    print(f"[DEBUG] match_node: matched_node_ids={matched_node_ids}, matched_names={matched_names}")
    return {"matched_node_ids": matched_node_ids, "matched_names": matched_names, **state}

def expand_graph(state):
    """Expand graph bi-directionally from matched node(s)."""
    from utils.neo4j_graphrag import Neo4jGraphRAG
    rag = Neo4jGraphRAG()
    matched_node_ids = state.get("matched_node_ids", [])
    print("[DEBUG] expand_graph: matched_node_ids =", matched_node_ids)
    if not matched_node_ids:
        # Fallback: try to match from input
        result = rag.query_flexible(state.get("input", ""))
        all_nodes = result.get("all_nodes", [])
    else:
        all_node_ids = rag.expand_from_nodes(matched_node_ids, max_nodes=25, direct_only=False)
        all_nodes = rag._get_node_names_from_ids(list(all_node_ids))
    print(f"[DEBUG] expand_graph: all_nodes={all_nodes}")
    return {"all_nodes": all_nodes, **state}

def generate_graph_html(state):
    from utils.neo4j_graphrag import Neo4jGraphRAG
    query = state.get("input", "")
    result = dict(state)
    print("[DEBUG] generate_graph_html: all_nodes =", result.get("all_nodes", []))
    rag = Neo4jGraphRAG()
    html = rag.generate_visualization_html(query, result)
    print("[DEBUG] generate_graph_html: graph_html generated, length=", len(html))
    print("[DEBUG] generate_graph_html: html content=", html)
    return {**state, "graph_html": html}

def extract_contexts(state):
    from utils.neo4j_graphrag import Neo4jGraphRAG
    rag = Neo4jGraphRAG()
    all_nodes = state.get("all_nodes", [])
    contexts = rag._extract_contexts_from_nodes(all_nodes)
    print(f"[DEBUG] extract_contexts: contexts={contexts}")
    return {**state, "contexts": contexts}

def format_response_node(state):
    from utils.neo4j_graphrag import format_response
    result = dict(state)
    answer = format_response(result)
    print(f"[DEBUG] format_response_node: answer={answer}")
    return {**state, "final_answer": answer}

def evaluate_ragas(state):
    try:
        from utils.ragas_evaluator import RAGASEvaluator
        import asyncio
        evaluator = RAGASEvaluator()
        answer = state.get("final_answer", "")
        contexts = state.get("contexts", [])
        user_query = state.get("input", "")
        ragas_scores = asyncio.run(
            evaluator.evaluate_single(user_query, answer, contexts, ground_truth=contexts[0] if contexts else "")
        )
        ragas_summary = evaluator.format_evaluation_summary(ragas_scores)
        print(f"[DEBUG] evaluate_ragas: ragas_scores={ragas_scores}")
        return {**state, "ragas_scores": ragas_scores, "ragas_summary": ragas_summary}
    except Exception as e:
        print(f"[DEBUG] evaluate_ragas exception: {e}")
        return {**state, "ragas_scores": {}, "ragas_summary": str(e)}

def return_to_ui(state):
    # Bundle everything for the frontend
    return {
        "graph_html": state.get("graph_html", ""),
        "final_answer": state.get("final_answer", ""),
        "ragas_summary": state.get("ragas_summary", ""),
        **state
    }
