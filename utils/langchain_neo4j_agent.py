from langchain_neo4j import Neo4jGraph
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from typing import Any, Generator, List, Dict
import re

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 1. Instantiate Neo4jGraph
neo4j_graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

def classify_query_intent(user_query: str, conversation_history: List[dict] = None) -> Dict[str, Any]:
    """
    Classify the user's query intent to determine the best search strategy.
    
    Returns:
        Dict with 'intent', 'confidence', 'search_strategy', and 'reasoning'
    """
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
        # Look for references to previous messages
        for turn in conversation_history[-3:]:  # Check last 3 turns
            if any(word in query_lower for word in turn["user_message"].lower().split()[:5]):
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
        # Default to graph search for ambiguous queries
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

def search_chat_history(query: str, conversation_history: List[dict], top_k: int = 3) -> List[dict]:
    """
    Search through chat history for relevant information.
    Uses simple keyword matching and relevance scoring.
    """
    if not conversation_history:
        return []
    
    # Simple keyword-based search
    query_words = set(query.lower().split())
    relevant_turns = []
    
    for turn in conversation_history:
        user_words = set(turn["user_message"].lower().split())
        assistant_words = set(turn["assistant_message"].lower().split())
        
        # Calculate relevance score
        user_overlap = len(query_words.intersection(user_words))
        assistant_overlap = len(query_words.intersection(assistant_words))
        total_overlap = user_overlap + assistant_overlap
        
        if total_overlap > 0:
            relevance_score = total_overlap / len(query_words)
            relevant_turns.append({
                "turn": turn,
                "relevance_score": relevance_score,
                "user_overlap": user_overlap,
                "assistant_overlap": assistant_overlap
            })
    
    # Sort by relevance and return top results
    relevant_turns.sort(key=lambda x: x["relevance_score"], reverse=True)
    return relevant_turns[:top_k]

def format_chat_history_results(relevant_turns: List[dict]) -> str:
    """Format chat history search results for the LLM."""
    if not relevant_turns:
        return "No relevant information found in conversation history."
    
    formatted = "Relevant information from our conversation history:\n\n"
    for i, result in enumerate(relevant_turns, 1):
        turn = result["turn"]
        score = result["relevance_score"]
        formatted += f"{i}. User: {turn['user_message']}\n"
        formatted += f"   Assistant: {turn['assistant_message']}\n"
        formatted += f"   (Relevance: {score:.2f})\n\n"
    
    return formatted

# 2. Create a proper tool wrapper for Neo4jGraph
class Neo4jQueryTool(BaseTool):
    name: str = "neo4j_query"
    description: str = """Execute Cypher queries against the Neo4j database to retrieve task and email data.

Available node types and their properties:
- Task: name, start_date, due_date, email_id, summary, topic, node_type
- Person: name, role, department, organization, relationship_type (owner/collaborator), node_type
- Topic: name, node_type
- Date: name, date_type (start/due), node_type (IMPORTANT: Use 'name' property, not 'date' property)
- Email_Index: name, node_type
- Summary: name, node_type
- Role: name, department, organization, node_type
- Department: name, organization, node_type
- Organization: name, node_type

Key relationships:
- (:Topic)-[:HAS_TASK]->(:Task)
- (:Task)-[:RESPONSIBLE_TO]->(:Person) - task owner
- (:Task)-[:COLLABORATED_BY]->(:Person) - task collaborators
- (:Task)-[:DUE_ON]->(:Date) - due dates
- (:Task)-[:START_ON]->(:Date) - start dates
- (:Person)-[:HAS_ROLE]->(:Role)-[:BELONGS_TO]->(:Department)-[:IS_IN]->(:Organization)

CRITICAL SEARCH STRATEGY - Search across ALL entities, not just task names:

1. **Multi-Entity Search**: When searching for terms, look in:
   - Task names, topics, summaries
   - Person names, roles, departments
   - Topic names
   - Email subjects and content
   - Organization names

2. **Use OR conditions** to search across multiple fields:
   ```cypher
   WHERE toLower(t.name) CONTAINS toLower('term') 
      OR toLower(t.topic) CONTAINS toLower('term')
      OR toLower(t.summary) CONTAINS toLower('term')
      OR toLower(p.name) CONTAINS toLower('term')
      OR toLower(p.role) CONTAINS toLower('term')
   ```

3. **Search in related entities** using OPTIONAL MATCH:
   ```cypher
   OPTIONAL MATCH (topic:Topic)-[:HAS_TASK]->(t2:Task)-[:RESPONSIBLE_TO]->(p2:Person)
   WHERE toLower(topic.name) CONTAINS toLower('term')
   ```

4. **Search in email content**:
   ```cypher
   OPTIONAL MATCH (email:Email_Index)-[:LINKED_TO]->(t3:Task)-[:RESPONSIBLE_TO]->(p3:Person)
   WHERE toLower(email.subject) CONTAINS toLower('term')
      OR toLower(email.content) CONTAINS toLower('term')
   ```

5. **Use DISTINCT and COALESCE** to combine results:
   ```cypher
   RETURN DISTINCT 
       COALESCE(p.name, p2.name, p3.name) as person_name,
       COALESCE(t.name, t2.name, t3.name) as task_name
   ```

Example comprehensive search queries:
- Find people involved in a project: 
  ```cypher
  MATCH (p:Person)-[:RESPONSIBLE_TO]->(t:Task)
  WHERE toLower(t.name) CONTAINS toLower('capstone') 
     OR toLower(t.topic) CONTAINS toLower('capstone')
     OR toLower(t.summary) CONTAINS toLower('capstone')
  OPTIONAL MATCH (topic:Topic)-[:HAS_TASK]->(t2:Task)-[:RESPONSIBLE_TO]->(p2:Person)
  WHERE toLower(topic.name) CONTAINS toLower('capstone')
  OPTIONAL MATCH (email:Email_Index)-[:LINKED_TO]->(t3:Task)-[:RESPONSIBLE_TO]->(p3:Person)
  WHERE toLower(email.subject) CONTAINS toLower('capstone')
     OR toLower(email.content) CONTAINS toLower('capstone')
  RETURN DISTINCT 
      COALESCE(p.name, p2.name, p3.name) as person_name,
      COALESCE(p.role, p2.role, p3.role) as person_role,
      COALESCE(t.name, t2.name, t3.name) as task_name
  ```

- Find tasks by topic or person:
  ```cypher
  MATCH (p:Person)-[:RESPONSIBLE_TO]->(t:Task)
  WHERE toLower(t.topic) CONTAINS toLower('project') 
     OR toLower(p.name) CONTAINS toLower('john')
     OR toLower(t.name) CONTAINS toLower('project')
  RETURN t.name, p.name, t.topic, t.due_date
  ```

- Comprehensive project search:
  ```cypher
  MATCH (p:Person)-[:RESPONSIBLE_TO]->(t:Task)
  WHERE toLower(t.name) CONTAINS toLower('term') 
     OR toLower(t.topic) CONTAINS toLower('term') 
     OR toLower(t.summary) CONTAINS toLower('term')
     OR toLower(p.name) CONTAINS toLower('term')
     OR toLower(p.role) CONTAINS toLower('term')
  OPTIONAL MATCH (topic:Topic)-[:HAS_TASK]->(t2:Task)-[:RESPONSIBLE_TO]->(p2:Person)
  WHERE toLower(topic.name) CONTAINS toLower('term')
  RETURN DISTINCT 
      COALESCE(p.name, p2.name) as person_name,
      COALESCE(p.role, p2.role) as person_role,
      COALESCE(t.name, t2.name) as task_name,
      COALESCE(t.topic, t2.topic) as topic_name
  ```

IMPORTANT: Always search across multiple entities and fields to ensure comprehensive results!"""
    
    def _run(self, query: str) -> dict:
        """Execute a Cypher query and return the results."""
        # Preprocessing: Remove code block formatting if present
        q = query.strip()
        if q.startswith('```'):
            # Remove triple backticks and optional 'cypher' language tag
            q = q.lstrip('`').strip()
            if q.lower().startswith('cypher'):
                q = q[6:].strip()
            # Remove trailing backticks if present
            if q.endswith('```'):
                q = q[:-3].strip()
        else:
            q = query
        try:
            result = neo4j_graph.query(q)
            return {"result": result, "query": q}
        except Exception as e:
            return {"error": str(e), "query": q}
    
    def _arun(self, query: str) -> str:
        """Async version of the tool."""
        return self._run(query)

# 3. Create the tool instance
neo4j_tool = Neo4jQueryTool()

# 4. Create a LangChain agent with the proper tool
llm = ChatOpenAI(temperature=0, model="gpt-4o")  # or gpt-4 if available
llm_stream = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

agent_executor = initialize_agent(
    tools=[neo4j_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

agent_executor_stream = initialize_agent(
    tools=[neo4j_tool],
    llm=llm_stream,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

def answer_with_neo4j_agent(user_query: str, conversation_history: List[dict] = None) -> dict:
    """
    Use the LangChain agent to answer a user query with smart fallback strategy.
    Now includes conversation history for context-aware responses and fallback to chat history.
    
    Args:
        user_query: The current user question
        conversation_history: List of previous conversation turns [{"user_message": "...", "assistant_message": "..."}]
    
    Returns a dict with 'output' (answer), 'query' (Cypher used), and 'strategy_used'.
    """
    # 1. Classify query intent
    intent_analysis = classify_query_intent(user_query, conversation_history)
    
    # 2. Build messages with conversation history
    messages = []
    if conversation_history:
        for turn in conversation_history[-5:]:  # Last 5 turns for context
            messages.append(HumanMessage(content=turn["user_message"]))
            messages.append(AIMessage(content=turn["assistant_message"]))
    
    # Add current query
    messages.append(HumanMessage(content=user_query))
    
    # 3. Execute search strategy
    if intent_analysis["search_strategy"] == "chat_history_first":
        # Try chat history first
        relevant_turns = search_chat_history(user_query, conversation_history or [])
        if relevant_turns and relevant_turns[0]["relevance_score"] > 0.3:
            # Use chat history results
            history_context = format_chat_history_results(relevant_turns)
            system_prompt = f"""You are a helpful assistant. Use the conversation history below to answer the user's question. If the history doesn't contain enough information, say so.

{history_context}

User Question: {user_query}"""
            
            response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}])
            return {
                "output": response.content,
                "query": None,
                "strategy_used": "chat_history",
                "intent_analysis": intent_analysis,
                "relevance_score": relevant_turns[0]["relevance_score"] if relevant_turns else 0
            }
    
    # 4. Try graph search (either as primary strategy or fallback)
    try:
        # Create a new agent instance with the conversation context
        context_agent = initialize_agent(
            tools=[neo4j_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        result = context_agent.invoke({"input": user_query, "chat_history": messages})
        
        # Check if graph search returned meaningful results
        graph_has_results = False
        cypher_query = None
        
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    if hasattr(action, "tool_input"):
                        cypher_query = action.tool_input
                    # Check if observation contains actual data
                    if isinstance(observation, dict) and "result" in observation:
                        if observation["result"] and len(observation["result"]) > 0:
                            graph_has_results = True
        
        # If graph search failed or returned no results, try chat history as fallback
        if not graph_has_results and conversation_history:
            relevant_turns = search_chat_history(user_query, conversation_history)
            if relevant_turns and relevant_turns[0]["relevance_score"] > 0.2:
                history_context = format_chat_history_results(relevant_turns)
                fallback_prompt = f"""The graph search didn't find relevant task data. However, I found some relevant information from our conversation history:

{history_context}

Based on this conversation history, please answer: {user_query}"""
                
                fallback_response = llm.invoke([{"role": "system", "content": fallback_prompt}, {"role": "user", "content": user_query}])
                return {
                    "output": fallback_response.content,
                    "query": cypher_query,
                    "strategy_used": "graph_then_chat_history_fallback",
                    "intent_analysis": intent_analysis,
                    "graph_has_results": graph_has_results
                }
        
        return {
            "output": result["output"],
            "query": cypher_query,
            "strategy_used": "graph_only",
            "intent_analysis": intent_analysis,
            "graph_has_results": graph_has_results
        }
        
    except Exception as e:
        # If graph search fails completely, try chat history
        if conversation_history:
            relevant_turns = search_chat_history(user_query, conversation_history)
            if relevant_turns:
                history_context = format_chat_history_results(relevant_turns)
                error_fallback_prompt = f"""The graph search encountered an error. However, I found some relevant information from our conversation history:

{history_context}

Based on this conversation history, please answer: {user_query}"""
                
                error_response = llm.invoke([{"role": "system", "content": error_fallback_prompt}, {"role": "user", "content": user_query}])
                return {
                    "output": error_response.content,
                    "query": None,
                    "strategy_used": "error_fallback_to_chat_history",
                    "intent_analysis": intent_analysis,
                    "error": str(e)
                }
        
        # If all else fails, return error
        return {
            "output": f"I encountered an error while searching for information: {str(e)}",
            "query": None,
            "strategy_used": "error",
            "intent_analysis": intent_analysis,
            "error": str(e)
        }

# Streaming version with smart fallback strategy
def stream_neo4j_agent(user_query: str, conversation_history: List[dict] = None) -> Generator[str, None, dict]:
    """
    Stream the LangChain agent's response with smart fallback strategy.
    Now includes conversation history for context-aware responses and fallback to chat history.
    
    Args:
        user_query: The current user question
        conversation_history: List of previous conversation turns [{"user_message": "...", "assistant_message": "..."}]
    
    Yields tokens as they arrive, then returns the final result dict.
    """
    # 1. Classify query intent
    intent_analysis = classify_query_intent(user_query, conversation_history)
    
    # 2. Build messages with conversation history
    messages = []
    if conversation_history:
        for turn in conversation_history[-5:]:  # Last 5 turns for context
            messages.append(HumanMessage(content=turn["user_message"]))
            messages.append(AIMessage(content=turn["assistant_message"]))
    
    # Add current query
    messages.append(HumanMessage(content=user_query))
    
    # 3. Execute search strategy
    if intent_analysis["search_strategy"] == "chat_history_first":
        # Try chat history first
        relevant_turns = search_chat_history(user_query, conversation_history or [])
        if relevant_turns and relevant_turns[0]["relevance_score"] > 0.3:
            # Use chat history results
            history_context = format_chat_history_results(relevant_turns)
            system_prompt = f"""You are a helpful assistant. Use the conversation history below to answer the user's question. If the history doesn't contain enough information, say so.

{history_context}

User Question: {user_query}"""
            
            # Stream the response
            buffer = ""
            for chunk in llm_stream.stream([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]):
                if hasattr(chunk, 'content'):
                    token = chunk.content
                    buffer += token
                    yield token
            
            return {
                "output": buffer,
                "query": None,
                "strategy_used": "chat_history",
                "intent_analysis": intent_analysis,
                "relevance_score": relevant_turns[0]["relevance_score"] if relevant_turns else 0
            }
    
    # 4. Try graph search (either as primary strategy or fallback)
    try:
        # Create a new streaming agent instance with the conversation context
        context_agent_stream = initialize_agent(
            tools=[neo4j_tool],
            llm=llm_stream,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # LangChain streaming returns a generator of events
        result = None
        buffer = ""
        for chunk in context_agent_stream.stream({"input": user_query, "chat_history": messages}):
            if "output" in chunk:
                token = chunk["output"]
                buffer += token
                yield token
            result = chunk
        
        # Check if graph search returned meaningful results
        graph_has_results = False
        cypher_query = None
        
        if result and "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    if hasattr(action, "tool_input"):
                        cypher_query = action.tool_input
                    # Check if observation contains actual data
                    if isinstance(observation, dict) and "result" in observation:
                        if observation["result"] and len(observation["result"]) > 0:
                            graph_has_results = True
        
        # If graph search failed or returned no results, try chat history as fallback
        if not graph_has_results and conversation_history:
            relevant_turns = search_chat_history(user_query, conversation_history)
            if relevant_turns and relevant_turns[0]["relevance_score"] > 0.2:
                history_context = format_chat_history_results(relevant_turns)
                fallback_prompt = f"""The graph search didn't find relevant task data. However, I found some relevant information from our conversation history:

{history_context}

Based on this conversation history, please answer: {user_query}"""
                
                # Stream the fallback response
                fallback_buffer = ""
                for chunk in llm_stream.stream([{"role": "system", "content": fallback_prompt}, {"role": "user", "content": user_query}]):
                    if hasattr(chunk, 'content'):
                        token = chunk.content
                        fallback_buffer += token
                        yield token
                
                return {
                    "output": fallback_buffer,
                    "query": cypher_query,
                    "strategy_used": "graph_then_chat_history_fallback",
                    "intent_analysis": intent_analysis,
                    "graph_has_results": graph_has_results
                }
        
        return {
            "output": buffer,
            "query": cypher_query,
            "strategy_used": "graph_only",
            "intent_analysis": intent_analysis,
            "graph_has_results": graph_has_results
        }
        
    except Exception as e:
        # If graph search fails completely, try chat history
        if conversation_history:
            relevant_turns = search_chat_history(user_query, conversation_history)
            if relevant_turns:
                history_context = format_chat_history_results(relevant_turns)
                error_fallback_prompt = f"""The graph search encountered an error. However, I found some relevant information from our conversation history:

{history_context}

Based on this conversation history, please answer: {user_query}"""
                
                # Stream the error fallback response
                error_buffer = ""
                for chunk in llm_stream.stream([{"role": "system", "content": error_fallback_prompt}, {"role": "user", "content": user_query}]):
                    if hasattr(chunk, 'content'):
                        token = chunk.content
                        error_buffer += token
                        yield token
                
                return {
                    "output": error_buffer,
                    "query": None,
                    "strategy_used": "error_fallback_to_chat_history",
                    "intent_analysis": intent_analysis,
                    "error": str(e)
                }
        
        # If all else fails, return error
        error_msg = f"I encountered an error while searching for information: {str(e)}"
        yield error_msg
        return {
            "output": error_msg,
            "query": None,
            "strategy_used": "error",
            "intent_analysis": intent_analysis,
            "error": str(e)
        }
