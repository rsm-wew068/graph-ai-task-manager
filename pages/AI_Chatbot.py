import streamlit as st
import uuid
 

try:
    from utils.langgraph_dag import run_agent_chat_round
    from utils.langchain_neo4j_agent import answer_with_neo4j_agent, stream_neo4j_agent
    from utils.langgraph_unified_memory_agent import (
        answer_with_simple_memory, 
        stream_simple_memory, 
        get_conversation_history, 
        clear_conversation
    )
    from utils.chainqa_graph_agent import chainqa_graph_search, stream_chainqa_graph_search
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ChatGPT-style styling
st.set_page_config(
    page_title="AI Task Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like appearance
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .main header {
        background: transparent;
        border-bottom: 1px solid #e5e5e5;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: transparent;
        border: none;
        padding: 1rem 0;
        margin: 0;
    }
    
    .stChatMessage[data-testid="chatMessage"] {
        border-radius: 0;
        border: none;
        box-shadow: none;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="chatMessage"] .stChatMessageContent {
        background: #f7f7f8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e5e5;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="chatMessage"] .stChatMessageContent {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e5e5;
    }
    
    /* Chat input styling */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background: white;
        border-top: 1px solid #e5e5e5;
        padding: 1rem;
        z-index: 1000;
    }
    
    .stChatInput .stTextInput {
        border-radius: 20px;
        border: 1px solid #e5e5e5;
        padding: 0.75rem 1rem;
        font-size: 16px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f7f7f8;
        border-right: 1px solid #e5e5e5;
        min-width: 250px;
    }
    
    /* Ensure sidebar is visible */
    section[data-testid="stSidebar"] {
        background: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #e5e5e5;
        background: white;
        color: #374151;
        padding: 0.5rem 1rem;
        font-size: 14px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #f3f4f6;
        border-color: #d1d5db;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 1.5s steps(5, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e5e5e5; margin-bottom: 2rem;">
    <h1 style="margin: 0; color: #374151; font-size: 2rem; font-weight: 600;">AI Task Assistant</h1>
    <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 1rem;">Ask me anything about your tasks, projects, or just chat!</p>
</div>
""", unsafe_allow_html=True)

# Show data status
if (hasattr(st.session_state, 'processing_complete') and st.session_state.processing_complete):
    outputs = st.session_state.get("extracted_tasks", [])
    valid_tasks_count = len([
        res for res in outputs
        if "validated_json" in res and res.get("valid", False)
    ])
    st.success(
        f"📊 Ready to answer questions about {valid_tasks_count} tasks "
        f"from your processed emails"
    )
    
    # Check Neo4j connection status
    try:
        from neo4j import GraphDatabase
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
        
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (t:Task) RETURN count(t) as count")
            task_count = result.single()["count"]
            st.success(f"🗄️ Connected to Neo4j database with {task_count} tasks")
        driver.close()
    except Exception as e:
        st.warning(f"⚠️ Neo4j connection issue: {str(e)}")
        
elif (hasattr(st.session_state, 'parsing_complete') and st.session_state.parsing_complete):
    st.warning(
        "📁 Emails parsed but not yet processed with LLM. "
        "Go to the main page to start LLM processing for Q&A."
    )
else:
    st.info(
        "📝 No data loaded yet. Please upload and process emails "
        "from the main page first to enable Q&A."
    )

# Persistent conversation_id
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# Initialize chat history in session state if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load history from simple memory system
if not st.session_state.chat_history:
    st.session_state.chat_history = get_conversation_history(st.session_state.conversation_id)

# Sidebar for controls (ChatGPT-style)
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 1px solid #e5e5e5; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #374151;">Chat Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear conversation button
    if st.button("🗑️ New Chat", use_container_width=True):
        if clear_conversation(st.session_state.conversation_id):
            st.session_state.chat_history = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.success("New chat started!")
            st.rerun()
        else:
            st.error("Failed to start new chat")
    
    # Show conversation info
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 1px solid #e5e5e5; margin-bottom: 1rem;">
        <h4 style="margin: 0; color: #374151;">Conversation Info</h4>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"**ID:** {st.session_state.conversation_id[:8]}...")
    st.write(f"**Messages:** {len(st.session_state.chat_history)}")
    
    # Show memory system info
    st.markdown("""
    <div style="padding: 1rem 0;">
        <h4 style="margin: 0; color: #374151;">Memory System</h4>
        <p style="font-size: 0.9rem; color: #6b7280; margin: 0.5rem 0;">
        ChatGPT-style interface with session persistence and smart query routing.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main chat interface
chat_container = st.container()

# Display chat history
with chat_container:
    if not st.session_state.chat_history:
        # Show welcome message for new chat
        st.markdown(
            """
<div style="text-align: center; padding: 3rem 1rem; color: #6b7280;">
    <h2 style="margin: 0 0 1rem 0; color: #374151;">How can I help you today?</h2>
    <p style="margin: 0; font-size: 1.1rem;">
        I can help you with tasks, projects, and anything in your email data!
    </p>
    <p style="margin: 1rem 0 0 0; font-size: 0.9rem; color: #9ca3af;">
        💡 Try asking: "What data do you have?" or "What can you help me with?"
    </p>
</div>
            """, unsafe_allow_html=True
        )
    else:
        # Display existing chat history
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["user"])
            with st.chat_message("assistant"):
                st.write(turn["assistant"])

# Chat input at the bottom
user_query = st.chat_input("Message AI Task Assistant...")

if user_query:
    # Add user message to chat
    with chat_container:
        st.chat_message("user").write(user_query)

        # Streaming assistant response
        assistant_placeholder = st.chat_message("assistant").empty()

        # Show typing indicator
        with assistant_placeholder:
            st.markdown('<span class="loading-dots">AI is thinking</span>', unsafe_allow_html=True)

        tokens = []
        strategy_used = None
        intent_analysis = None

        def is_conversation_history_query(query: str) -> bool:
            import re
            conversation_patterns = [
                r'\b(what did|what was|what were|what have|what has)\b',
                r'\b(earlier|before|previously|last time|yesterday)\b',
                r'\b(conversation|chat|talk|discussion)\b',
                r'\b(you said|you mentioned|you told)\b',
                r'\b(remember|recall|remind)\b',
                r'\b(we discussed|we talked about|we covered)\b',
                r'\b(follow up|follow-up|followup)\b',
                r'\b(continue|continue from|pick up)\b',
                r'\b(what did i just|what did you just|what was my last|what was your last)\b',
                r'\b(just asked|just said|just mentioned)\b',
                r'\b(repeat|say again|tell me again)\b'
            ]
            query_lower = query.lower()
            return any(re.search(pattern, query_lower) for pattern in conversation_patterns)

        try:
            # Enhanced routing: check for task-specific queries first
            task_keywords = ['task', 'tasks', 'project', 'projects', 'due', 'deadline', 'responsible', 'assign', 'work', 'todo', 'action']
            is_task_query = any(keyword in user_query.lower() for keyword in task_keywords)
            
            # Check for help queries
            help_keywords = ['help', 'what can you', 'what do you know', 'what data', 'what information', 'show me what', 'what is available']
            is_help_query = any(keyword in user_query.lower() for keyword in help_keywords)
            
            assistant_message = ""  # Always define as string
            if is_help_query:
                # Show what data is available in the database
                try:
                    from neo4j import GraphDatabase
                    import os
                    from dotenv import load_dotenv
                    load_dotenv()
                    
                    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
                    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
                    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
                    
                    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                    
                    with driver.session() as session:
                        # Get database overview
                        result = session.run("""
                            MATCH (n)
                            RETURN labels(n) as labels, count(n) as count
                            ORDER BY count DESC
                        """)
                        node_counts = list(result)
                        
                        # Get sample tasks
                        result = session.run("MATCH (t:Task) RETURN t.name as name LIMIT 5")
                        sample_tasks = list(result)
                        
                        # Get sample people
                        result = session.run("MATCH (p:Person) RETURN p.name as name LIMIT 5")
                        sample_people = list(result)
                        
                        assistant_message = "**Here's what I can help you with based on your data:**\n\n"
                        
                        if node_counts:
                            assistant_message += "**📊 Database Overview:**\n"
                            for item in node_counts:
                                labels = item['labels'][0] if item['labels'] else 'Unknown'
                                assistant_message += f"• {labels}: {item['count']} items\n"
                        
                        if sample_tasks:
                            assistant_message += f"\n**📋 Sample Tasks:**\n"
                            for task in sample_tasks:
                                assistant_message += f"• {task['name']}\n"
                        
                        if sample_people:
                            assistant_message += f"\n**👥 Sample People:**\n"
                            for person in sample_people:
                                assistant_message += f"• {person['name']}\n"
                        
                        assistant_message += "\n**💡 You can ask me questions like:**\n"
                        assistant_message += "• \"What tasks do I have?\"\n"
                        assistant_message += "• \"Who is responsible for [specific task]?\"\n"
                        assistant_message += "• \"What's due this week?\"\n"
                        assistant_message += "• \"Show me all my projects\"\n"
                        assistant_message += "• \"What did we discuss earlier?\"\n"
                        
                        strategy_used = "help_query"
                        intent_analysis = {"intent": "help", "confidence": 0.9}
                    
                    driver.close()
                    
                except Exception as e:
                    assistant_message = f"I encountered an error while checking your data: {str(e)}"
                    strategy_used = "help_query_error"
                    intent_analysis = {"intent": "error", "confidence": 0.0}
                    
            elif is_task_query:
                # Use ChainQA for all task-related questions (dynamic approach)
                try:
                    chainqa_result = chainqa_graph_search(user_query)
                    if chainqa_result.get("success"):
                        assistant_message = str(chainqa_result.get("answer", ""))
                        strategy_used = "chainqa_graph_search"
                        intent_analysis = {"intent": "task_data", "confidence": 0.9}
                    else:
                        assistant_message = f"I couldn't find specific information about that in your task data. {chainqa_result.get('answer', '')}"
                        strategy_used = "chainqa_graph_search_no_results"
                        intent_analysis = {"intent": "task_data", "confidence": 0.3}
                except Exception as e:
                    assistant_message = f"I encountered an error while querying your task data: {str(e)}. Please try rephrasing your question."
                    strategy_used = "chainqa_graph_search_error"
                    intent_analysis = {"intent": "error", "confidence": 0.0}
            elif is_conversation_history_query(user_query):
                # Use simple memory for conversation-history queries
                try:
                    # First, get the conversation history
                    history = get_conversation_history(st.session_state.conversation_id)
                    
                    if history:
                        # Show recent conversation history
                        assistant_message = "Here's what we've discussed recently:\n\n"
                        for i, turn in enumerate(history[-3:], 1):  # Show last 3 turns
                            assistant_message += f"**Turn {i}:**\n"
                            assistant_message += f"**You:** {turn['user']}\n"
                            assistant_message += f"**Me:** {turn['assistant']}\n\n"
                        
                        if "what did i just ask" in user_query.lower():
                            if history:
                                last_user_message = history[-1]['user']
                                assistant_message = f"Your last question was: **\"{last_user_message}\"**"
                            else:
                                assistant_message = "I don't see any previous questions in our conversation."
                    else:
                        assistant_message = "We haven't had any previous conversation yet. This is our first exchange!"
                    
                    strategy_used = "conversation_history"
                    intent_analysis = {"intent": "conversation_history", "confidence": 0.9}
                    
                except Exception as e:
                    assistant_message = f"I encountered an error while retrieving our conversation history: {str(e)}"
                    strategy_used = "conversation_history_error"
                    intent_analysis = {"intent": "error", "confidence": 0.0}
            elif any(keyword in user_query.lower() for keyword in ['who', 'when', 'what', 'where', 'project', 'due', 'responsible', 'working']):
                # Use ChainQA for graph-related queries (non-streaming version)
                chainqa_result = {}
                temp_val3 = chainqa_graph_search(user_query)
                if isinstance(temp_val3, dict):
                    chainqa_result = temp_val3
                assistant_placeholder.empty()
                if chainqa_result.get("success"):
                    assistant_message = str(chainqa_result.get("answer", ""))
                    assistant_placeholder.write(assistant_message)
                    strategy_used = "chainqa_graph_search"
                    intent_analysis = {"intent": "graph_data", "confidence": 0.9}
                elif chainqa_result:
                    error_message = f"I encountered an error while processing your request: {chainqa_result.get('answer', 'Unknown error')}"
                    assistant_message = error_message
                    assistant_placeholder.write(error_message)
                    strategy_used = "chainqa_graph_search_error"
                    intent_analysis = {"intent": "error", "confidence": 0.0}
                else:
                    error_message = "I encountered an unknown error while processing your request."
                    assistant_message = error_message
                    assistant_placeholder.write(error_message)
                    strategy_used = "chainqa_graph_search_error"
                    intent_analysis = {"intent": "error", "confidence": 0.0}
            else:
                # Use simple memory for other conversation queries
                try:
                    # Use the simple memory system for general conversation
                    stream = stream_simple_memory(user_query, "user123", st.session_state.conversation_id)
                    assistant_placeholder.empty()
                    for token in stream:
                        tokens.append(token)
                        assistant_placeholder.write("".join(tokens))
                    memory_result2 = {}
                    try:
                        temp_val4 = stream.send(None)
                        if isinstance(temp_val4, dict):
                            memory_result2 = temp_val4
                    except StopIteration as e:
                        temp_val5 = getattr(e, 'value', None)
                        if isinstance(temp_val5, dict):
                            memory_result2 = temp_val5
                    strategy_used = memory_result2.get("search_strategy", "conversation_only")
                    intent_analysis = memory_result2.get("intent_analysis", {})
                    assistant_message = "".join(tokens)
                except Exception as e:
                    assistant_message = f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."
                    strategy_used = "conversation_error"
                    intent_analysis = {"intent": "error", "confidence": 0.0}

            # Update chat history
            st.session_state.chat_history.append({
                "user": user_query,
                "assistant": assistant_message
            })

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            error_message = f"""I encountered an error while processing your request: {str(e)}

This might be due to:
- Database connection issues
- Query syntax problems
- Missing data

Please try rephrasing your question or ask about something else."""
            assistant_placeholder.write(error_message)

    # Show strategy information in a subtle way
    if strategy_used and strategy_used != "unknown":
        with st.expander("🔍 Search Details", expanded=False):
            st.write(f"**Strategy:** {strategy_used}")
            if intent_analysis:
                st.write(f"**Intent:** {intent_analysis.get('intent', 'unknown')}")
                st.write(f"**Confidence:** {intent_analysis.get('confidence', 0):.2f}")

# Footer with helpful tips
if not st.session_state.chat_history:
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e5e5;">
        <h3 style="margin: 0 0 1rem 0; color: #374151;">💡 Try asking me about:</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <h4 style="margin: 0 0 0.5rem 0; color: #374151;">🗂️ Tasks & Projects</h4>
                <ul style="margin: 0; padding-left: 1.5rem; color: #6b7280;">
                    <li>"What tasks are due this week?"</li>
                    <li>"Who is working on the Capstone Project?"</li>
                    <li>"Show me all project management tasks"</li>
                </ul>
            </div>
            <div>
                <h4 style="margin: 0 0 0.5rem 0; color: #374151;">💬 General Chat</h4>
                <ul style="margin: 0; padding-left: 1.5rem; color: #6b7280;">
                    <li>"What did we discuss earlier?"</li>
                    <li>"Can you remind me about deadlines?"</li>
                    <li>"Tell me more about that project"</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)