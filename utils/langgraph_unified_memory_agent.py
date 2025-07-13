"""
Simplified LangGraph memory system using only short-term memory with session persistence.
This provides a ChatGPT-style interface where conversation history persists during the session.
"""

from typing import Any, Dict, List, Generator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
import re
import uuid
import json
from datetime import datetime

load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize Neo4j graph
neo4j_graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# Initialize short-term memory with session persistence
checkpointer = InMemorySaver()

# LLM models
llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm_stream = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

class AppState(MessagesState):
    """Extended state to include conversation metadata and search results."""
    conversation_id: str
    user_id: str
    search_results: List[Dict[str, Any]] = []
    search_strategy: str = "graph_first"
    intent_analysis: Dict[str, Any] = {}

def search_conversation_history(query: str, conversation_history: List[BaseMessage], top_k: int = 3) -> List[Dict[str, Any]]:
    """Search through conversation history for relevant information."""
    if not conversation_history:
        return []
    
    query_words = set(query.lower().split())
    relevant_turns = []
    
    # Process messages in pairs (user + assistant)
    for i in range(0, len(conversation_history) - 1, 2):
        if i + 1 < len(conversation_history):
            user_msg = conversation_history[i]
            assistant_msg = conversation_history[i + 1]
            
            if user_msg.type == "human" and assistant_msg.type == "ai":
                user_words = set(user_msg.content.lower().split())
                assistant_words = set(assistant_msg.content.lower().split())
                
                # Calculate relevance score
                user_overlap = len(query_words.intersection(user_words))
                assistant_overlap = len(query_words.intersection(assistant_words))
                total_overlap = user_overlap + assistant_overlap
                
                if total_overlap > 0:
                    relevance_score = total_overlap / len(query_words)
                    relevant_turns.append({
                        "user_message": user_msg.content,
                        "assistant_message": assistant_msg.content,
                        "relevance_score": relevance_score
                    })
    
    # Sort by relevance and return top results
    relevant_turns.sort(key=lambda x: x["relevance_score"], reverse=True)
    return relevant_turns[:top_k]

def classify_query_intent(user_query: str, conversation_history: List[BaseMessage]) -> Dict[str, Any]:
    """Classify query intent using conversation history."""
    query_lower = user_query.lower()
    
    # Patterns that indicate conversation history queries
    conversation_patterns = [
        r'\b(what did|what was|what were|what have|what has)\b',
        r'\b(earlier|before|previously|last time|yesterday)\b',
        r'\b(conversation|chat|talk|discussion)\b',
        r'\b(you said|you mentioned|you told)\b',
        r'\b(remember|recall|remind)\b',
        r'\b(we discussed|we talked about|we covered)\b',
        r'\b(follow up|follow-up|followup)\b',
        r'\b(continue|continue from|pick up)\b'
    ]
    
    # Patterns that indicate task/data queries
    task_patterns = [
        r'\b(task|tasks|project|projects|deadline|deadlines)\b',
        r'\b(due|due date|start date|timeline)\b',
        r'\b(who is|who are|responsible|assigned)\b',
        r'\b(show me|find|search|list)\b',
        r'\b(how many|count|total)\b',
        r'\b(status|progress|update)\b',
        r'\b(collaborator|team member|person)\b',
        r'\b(topic|category|type)\b'
    ]
    
    # Check for conversation patterns
    conversation_matches = sum(1 for pattern in conversation_patterns if re.search(pattern, query_lower))
    task_matches = sum(1 for pattern in task_patterns if re.search(pattern, query_lower))
    
    # Check if query references previous conversation
    history_reference = False
    if conversation_history:
        for msg in conversation_history[-6:]:  # Check last 6 messages (3 turns)
            if msg.type == "human" and any(word in query_lower for word in msg.content.lower().split()[:5]):
                history_reference = True
                break
    
    # Determine intent
    if conversation_matches > task_matches or history_reference:
        intent = "conversation_history"
        confidence = 0.8 if history_reference else 0.6
    else:
        intent = "task_data"
        confidence = 0.7
    
    return {
        "intent": intent,
        "confidence": confidence,
        "conversation_matches": conversation_matches,
        "task_matches": task_matches,
        "history_reference": history_reference
    }

class Neo4jQueryTool(BaseTool):
    name: str = "neo4j_query"
    description: str = """Execute Cypher queries against the Neo4j database to retrieve task and email data.
    
    Use this tool to search for:
    - Tasks and their details
    - Email information
    - People and organizations
    - Topics and categories
    - Relationships between entities
    
    Always search across multiple entity types using OR conditions and OPTIONAL MATCH to ensure comprehensive results.
    """
    
    def _run(self, query: str) -> str:
        try:
            # Clean the query - remove markdown formatting if present
            cleaned_query = query.strip()
            if cleaned_query.startswith("```cypher"):
                cleaned_query = cleaned_query[9:]  # Remove ```cypher
            if cleaned_query.startswith("```"):
                cleaned_query = cleaned_query[3:]  # Remove ```
            if cleaned_query.endswith("```"):
                cleaned_query = cleaned_query[:-3]  # Remove ```
            cleaned_query = cleaned_query.strip()
            
            result = neo4j_graph.query(cleaned_query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def _arun(self, query: str) -> str:
        return self._run(query)

def pre_model_hook(state: AppState):
    """Trim messages to prevent context overflow."""
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=4000,  # Increased for better context
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}

def classify_and_search_node(state: AppState) -> Dict[str, Any]:
    """Classify query intent and search appropriate data sources."""
    user_query = state["messages"][-1].content if state["messages"] else ""
    conversation_history = state["messages"][:-1] if len(state["messages"]) > 1 else []
    
    # Classify the query intent
    intent_analysis = classify_query_intent(user_query, conversation_history)
    state["intent_analysis"] = intent_analysis
    
    search_results = []
    search_strategy = "conversation_only"
    
    if intent_analysis["intent"] == "task_data":
        # Try to search Neo4j graph data
        try:
            # Create a comprehensive search query that's more robust
            # Escape single quotes and handle special characters
            safe_query = user_query.replace("'", "\\'").replace('"', '\\"')
            
            # Use only properties that exist in the database
            search_query = f"""
            MATCH (n)
            WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower('{safe_query}'))
               OR (n.summary IS NOT NULL AND toLower(n.summary) CONTAINS toLower('{safe_query}'))
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN DISTINCT n, r, related
            LIMIT 10
            """
            
            result = neo4j_graph.query(search_query)
            
            if result and len(result) > 0:
                search_results = result
                search_strategy = "graph_data"
            else:
                # Fallback to conversation history
                search_results = search_conversation_history(user_query, conversation_history)
                search_strategy = "conversation_fallback"
                
        except Exception as e:
            print(f"Neo4j search error: {e}")
            # Fallback to conversation history
            search_results = search_conversation_history(user_query, conversation_history)
            search_strategy = "conversation_fallback"
    else:
        # Search conversation history
        search_results = search_conversation_history(user_query, conversation_history)
        search_strategy = "conversation_only"
    
    state["search_results"] = search_results
    state["search_strategy"] = search_strategy
    
    return {
        "search_results": search_results,
        "search_strategy": search_strategy,
        "intent_analysis": intent_analysis
    }

def generate_response_node(state: AppState) -> Dict[str, Any]:
    """Generate response based on search results and conversation context."""
    user_query = state["messages"][-1].content if state["messages"] else ""
    search_results = state["search_results"]
    search_strategy = state["search_strategy"]
    intent_analysis = state["intent_analysis"]
    
    # Build context from search results
    context = ""
    if search_results:
        if search_strategy.startswith("graph"):
            context = f"Relevant data from database: {json.dumps(search_results, indent=2)}"
        else:
            context = "Relevant conversation history:\n"
            for result in search_results:
                context += f"User: {result['user_message']}\nAssistant: {result['assistant_message']}\n\n"
    
    # Create system message
    system_message = f"""You are a helpful AI assistant. 

Search Strategy: {search_strategy}
Query Intent: {intent_analysis['intent']} (confidence: {intent_analysis['confidence']})

{context}

Provide a helpful response based on the available information. If searching the database, focus on tasks, emails, and related data. If searching conversation history, reference previous discussions appropriately."""

    # Trim messages to prevent context overflow
    messages_to_use = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=4000,
        start_on="human",
        end_on=("human", "tool"),
    )
    
    # Generate response
    messages = [{"role": "system", "content": system_message}] + [
        {"role": msg.type, "content": msg.content} 
        for msg in messages_to_use
    ]
    
    response = llm.invoke(messages)
    
    return {"messages": [response]}

def create_simple_memory_agent():
    """Create a simple agent with short-term memory only."""
    builder = StateGraph(AppState)
    
    # Add nodes
    builder.add_node("classify_and_search", classify_and_search_node)
    builder.add_node("generate_response", generate_response_node)
    
    # Add edges
    builder.add_edge(START, "classify_and_search")
    builder.add_edge("classify_and_search", "generate_response")
    
    # Compile with checkpointer for session persistence
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph

def answer_with_simple_memory(user_query: str, user_id: str, conversation_id: str = None) -> Dict[str, Any]:
    """Answer a query using simple short-term memory."""
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    config = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": user_id
        }
    }
    
    graph = create_simple_memory_agent()
    
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "conversation_id": conversation_id,
            "user_id": user_id
        },
        config=config
    )
    
    return {
        "response": result["messages"][-1].content,
        "conversation_id": conversation_id,
        "search_strategy": result.get("search_strategy", "unknown"),
        "intent_analysis": result.get("intent_analysis", {})
    }

def stream_simple_memory(user_query: str, user_id: str, conversation_id: str = None) -> Generator[str, None, Dict[str, Any]]:
    """Stream response using simple short-term memory."""
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    config = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": user_id
        }
    }
    
    graph = create_simple_memory_agent()
    
    # First, get the search results and context
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "conversation_id": conversation_id,
            "user_id": user_id
        },
        config=config
    )
    
    # Extract the final response and metadata
    final_response = result["messages"][-1].content if result["messages"] else ""
    metadata = {
        "conversation_id": conversation_id,
        "search_strategy": result.get("search_strategy", "unknown"),
        "intent_analysis": result.get("intent_analysis", {})
    }
    
    # Stream the response token by token
    words = final_response.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
    
    # Return metadata at the end
    return metadata

def get_conversation_history(conversation_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session."""
    config = {"configurable": {"thread_id": conversation_id}}
    graph = create_simple_memory_agent()
    
    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        
        history = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                history.append({
                    "user": messages[i].content,
                    "assistant": messages[i + 1].content
                })
        
        return history
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

def clear_conversation(conversation_id: str):
    """Clear conversation history for a session."""
    config = {"configurable": {"thread_id": conversation_id}}
    
    try:
        # Delete the checkpoint to clear conversation history
        checkpointer.delete(config)
        return True
    except Exception as e:
        print(f"Error clearing conversation: {e}")
        return False

def demonstrate_simple_memory():
    """Demonstrate the simple memory system."""
    conversation_id = str(uuid.uuid4())
    user_id = "demo_user"
    
    print("=== Simple Memory System Demo ===\n")
    
    # First query
    print("1. First query:")
    result1 = answer_with_simple_memory(
        "What tasks do I have due this week?",
        user_id,
        conversation_id
    )
    print(f"Response: {result1['response']}")
    print(f"Strategy: {result1['search_strategy']}")
    print()
    
    # Second query (should reference conversation history)
    print("2. Follow-up query:")
    result2 = answer_with_simple_memory(
        "Can you remind me what we just discussed?",
        user_id,
        conversation_id
    )
    print(f"Response: {result2['response']}")
    print(f"Strategy: {result2['search_strategy']}")
    print()
    
    # Show conversation history
    print("3. Conversation History:")
    history = get_conversation_history(conversation_id)
    for i, turn in enumerate(history, 1):
        print(f"Turn {i}:")
        print(f"  User: {turn['user']}")
        print(f"  Assistant: {turn['assistant']}")
        print()

if __name__ == "__main__":
    demonstrate_simple_memory() 