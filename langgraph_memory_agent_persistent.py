"""
Example of extending LangGraph memory system with persistent storage for truly long-term memory.
This shows how to use PostgreSQL or other persistent stores for cross-session memory.
"""

from typing import Any, Dict, List, Generator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
import os
from dotenv import load_dotenv
import re
import uuid

load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# PostgreSQL configuration for persistent long-term memory
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/langgraph_memory")

# Initialize Neo4j graph
neo4j_graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# Initialize LangGraph memory systems with persistent storage
checkpointer = MemorySaver()  # Short-term memory (session-based)

# Long-term memory with PostgreSQL (truly persistent)
try:
    store = PostgresStore.from_conn_string(POSTGRES_URI)
    print("✅ Using PostgreSQL for persistent long-term memory")
except Exception as e:
    print(f"⚠️ PostgreSQL not available, falling back to in-memory: {e}")
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore()

# LLM models
llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm_stream = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

class AppState(MessagesState):
    """Extended state to include conversation metadata and search results."""
    conversation_id: str
    user_id: str  # For cross-conversation memory
    search_results: List[Dict[str, Any]] = []
    search_strategy: str = "graph_first"
    intent_analysis: Dict[str, Any] = {}

def store_user_memory(user_id: str, memory_key: str, memory_data: Dict[str, Any], store: BaseStore):
    """
    Store user-specific information in long-term memory.
    This persists across conversations and app restarts.
    """
    namespace = ("user_memories", user_id)
    store.put(namespace, memory_key, memory_data)

def retrieve_user_memories(user_id: str, query: str, store: BaseStore, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve user-specific memories from long-term storage.
    This can access information from previous conversations.
    """
    namespace = ("user_memories", user_id)
    try:
        memories = store.search(namespace, query=query, limit=limit)
        return [memory.value for memory in memories]
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return []

def classify_query_intent(user_query: str, messages: List[BaseMessage]) -> Dict[str, Any]:
    """Classify query intent using conversation history from LangGraph state."""
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
    
    # Check if query references previous conversation using LangGraph messages
    history_reference = False
    if len(messages) > 1:
        # Look for references to previous messages in the last few turns
        recent_messages = messages[-6:]  # Last 3 turns (user + assistant)
        for msg in recent_messages:
            if hasattr(msg, 'content') and msg.content:
                if any(word in query_lower for word in msg.content.lower().split()[:5]):
                    history_reference = True
                    break
    
    # Determine intent
    if conversation_matches > task_matches or history_reference:
        intent = "conversation_history"
        confidence = min(0.9, 0.6 + (conversation_matches * 0.1))
        strategy = "chat_history_first"
        reasoning = f"Query matches {conversation_matches} conversation patterns and {task_matches} task patterns"
    elif task_matches > 0:
        intent = "task_data"
        confidence = min(0.9, 0.6 + (task_matches * 0.1))
        strategy = "graph_first"
        reasoning = f"Query matches {task_matches} task patterns and {conversation_matches} conversation patterns"
    else:
        intent = "ambiguous"
        confidence = 0.5
        strategy = "graph_first"
        reasoning = "Query doesn't clearly match conversation or task patterns"
    
    return {
        "intent": intent,
        "confidence": confidence,
        "search_strategy": strategy,
        "reasoning": reasoning,
        "conversation_matches": conversation_matches,
        "task_matches": task_matches,
        "history_reference": history_reference
    }

def search_chat_history(query: str, messages: List[BaseMessage], top_k: int = 3) -> List[Dict[str, Any]]:
    """Search through LangGraph message history for relevant information."""
    if len(messages) < 2:
        return []
    
    query_words = set(query.lower().split())
    relevant_turns = []
    
    # Process messages in pairs (user + assistant)
    for i in range(0, len(messages) - 1, 2):
        if i + 1 < len(messages):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            if hasattr(user_msg, 'content') and hasattr(assistant_msg, 'content'):
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
                        "relevance_score": relevance_score,
                        "user_overlap": user_overlap,
                        "assistant_overlap": assistant_overlap
                    })
    
    # Sort by relevance and return top results
    relevant_turns.sort(key=lambda x: x["relevance_score"], reverse=True)
    return relevant_turns[:top_k]

class Neo4jQueryTool(BaseTool):
    name: str = "neo4j_query"
    description: str = """Execute Cypher queries against the Neo4j database to retrieve task and email data."""

    def _run(self, query: str) -> str:
        try:
            result = neo4j_graph.query(query)
            return str(result)
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def _arun(self, query: str) -> str:
        return self._run(query)

# Create LangChain agent
neo4j_tool = Neo4jQueryTool()
agent_executor = initialize_agent(
    tools=[neo4j_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

def pre_model_hook(state: AppState):
    """
    Pre-model hook for message trimming - follows LangGraph best practices.
    This keeps the original message history unmodified and only trims for LLM input.
    """
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}

def classify_and_search_node(state: AppState, *, store: BaseStore) -> Dict[str, Any]:
    """
    Node that classifies query intent and performs appropriate search.
    Uses LangGraph's store for long-term memory if needed.
    """
    # Get the last user message
    last_message = state["messages"][-1]
    user_query = last_message.content
    user_id = state.get("user_id", "default_user")
    
    # Classify query intent
    intent_analysis = classify_query_intent(user_query, state["messages"])
    state["intent_analysis"] = intent_analysis
    
    # Check long-term memory first for user-specific information
    long_term_memories = retrieve_user_memories(user_id, user_query, store)
    
    # Determine search strategy
    if intent_analysis["search_strategy"] == "chat_history_first":
        # Search conversation history first
        relevant_turns = search_chat_history(user_query, state["messages"])
        if relevant_turns:
            # Found relevant history, use that
            context = "Based on our conversation history:\n\n"
            for turn in relevant_turns:
                context += f"User: {turn['user_message']}\n"
                context += f"Assistant: {turn['assistant_message']}\n\n"
            
            # Add long-term memory if available
            if long_term_memories:
                context += "From our previous conversations:\n\n"
                for memory in long_term_memories:
                    context += f"- {memory.get('content', '')}\n"
                context += "\n"
            
            return {
                "search_results": relevant_turns,
                "search_strategy": "chat_history_first",
                "context": context
            }
    
    # Default to graph search
    try:
        # Use the agent to query Neo4j
        result = agent_executor.invoke({"input": user_query})
        graph_results = [{"type": "graph", "content": result["output"]}]
        
        context = f"Based on the task database: {result['output']}"
        
        # Add long-term memory if available
        if long_term_memories:
            context += "\n\nFrom our previous conversations:\n\n"
            for memory in long_term_memories:
                context += f"- {memory.get('content', '')}\n"
        
        return {
            "search_results": graph_results,
            "search_strategy": "graph_first",
            "context": context
        }
    except Exception as e:
        # Fallback to conversation history if graph search fails
        relevant_turns = search_chat_history(user_query, state["messages"])
        context = "I couldn't find relevant information in the task database. "
        if relevant_turns:
            context += "However, based on our conversation history:\n\n"
            for turn in relevant_turns:
                context += f"User: {turn['user_message']}\n"
                context += f"Assistant: {turn['assistant_message']}\n\n"
        
        # Add long-term memory if available
        if long_term_memories:
            context += "From our previous conversations:\n\n"
            for memory in long_term_memories:
                context += f"- {memory.get('content', '')}\n"
        
        return {
            "search_results": relevant_turns or [],
            "search_strategy": "fallback",
            "context": context
        }

def generate_response_node(state: AppState) -> Dict[str, Any]:
    """
    Node that generates the final response using the search results.
    """
    last_message = state["messages"][-1]
    user_query = last_message.content
    context = state.get("context", "")
    search_strategy = state.get("search_strategy", "graph_first")
    
    # Build the prompt
    if context:
        prompt = f"""You are a helpful AI assistant for task management. 

{context}

Current user question: {user_query}

Please provide a helpful and accurate response based on the information above. If you're using conversation history, acknowledge that you're referring to previous discussion."""
    else:
        prompt = f"Please answer this question about task management: {user_query}"
    
    # Generate response
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [response]}

# Build the LangGraph workflow
def create_persistent_memory_agent():
    """Create the LangGraph agent with persistent memory management."""
    
    # Create the graph
    workflow = StateGraph(AppState)
    
    # Add nodes
    workflow.add_node("classify_and_search", classify_and_search_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Add pre-model hook for message trimming
    workflow.add_node("pre_model_hook", pre_model_hook)
    
    # Set up the flow
    workflow.set_entry_point("classify_and_search")
    workflow.add_edge("classify_and_search", "pre_model_hook")
    workflow.add_edge("pre_model_hook", "generate_response")
    workflow.set_finish_point("generate_response")
    
    # Compile with memory systems
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    return graph

# Example usage functions
def answer_with_persistent_memory(user_query: str, user_id: str, conversation_id: str = None) -> Dict[str, Any]:
    """
    Use the LangGraph persistent memory agent to answer a user query.
    This can access information from previous conversations across app restarts.
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Prepare the input
    messages = [HumanMessage(content=user_query)]
    
    # Configure the thread
    config = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": user_id
        }
    }
    
    # Create the agent
    agent = create_persistent_memory_agent()
    
    # Invoke the graph
    result = agent.invoke(
        {"messages": messages, "conversation_id": conversation_id, "user_id": user_id},
        config=config
    )
    
    # Extract the response
    response_message = result["messages"][-1]
    
    return {
        "output": response_message.content,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "search_strategy": result.get("search_strategy", "unknown"),
        "intent_analysis": result.get("intent_analysis", {}),
        "search_results": result.get("search_results", [])
    }

def store_important_memory(user_id: str, memory_content: str, memory_type: str = "general"):
    """
    Store important information in long-term memory for future reference.
    This persists across conversations and app restarts.
    """
    memory_data = {
        "content": memory_content,
        "type": memory_type,
        "timestamp": "now"
    }
    store_user_memory(user_id, str(uuid.uuid4()), memory_data, store)
    print(f"✅ Stored memory for user {user_id}: {memory_content[:50]}...")

# Example of memory duration
def demonstrate_memory_duration():
    """
    Demonstrate the different memory durations in the system.
    """
    print("🧠 Memory Duration Demonstration")
    print("=" * 50)
    
    user_id = "demo_user_001"
    
    print("1️⃣ **Short-term Memory (Session-based)**")
    print("   - Duration: Until app restart or manual clear")
    print("   - Storage: MemorySaver (in-memory checkpointing)")
    print("   - Example: Chat for 30 minutes, refresh page → history preserved")
    print("   - Example: Restart app → history lost")
    print()
    
    print("2️⃣ **Long-term Memory (Cross-conversation)**")
    print("   - Duration: Indefinite (persists across app restarts)")
    print("   - Storage: PostgreSQL (or other persistent store)")
    print("   - Example: Start new conversation, ask about previous → can access old info")
    print("   - Example: Restart app → long-term memories still available")
    print()
    
    print("3️⃣ **Memory Storage Example**")
    # Store some important information
    store_important_memory(user_id, "User prefers to be called 'Alex' and works in the marketing department")
    store_important_memory(user_id, "User has a Capstone Project due on March 15th")
    store_important_memory(user_id, "User mentioned they prefer email summaries over detailed reports")
    
    print("4️⃣ **Memory Retrieval Example**")
    memories = retrieve_user_memories(user_id, "Capstone Project", store)
    print(f"Found {len(memories)} memories about Capstone Project:")
    for memory in memories:
        print(f"  - {memory.get('content', '')}")
    
    print()
    print("✅ This demonstrates how long-term memory persists across conversations!")

if __name__ == "__main__":
    demonstrate_memory_duration() 