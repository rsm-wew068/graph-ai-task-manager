import streamlit as st
import uuid
import requests

try:
    from utils.langgraph_dag import run_agent_chat_round
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(page_title="🤖 Ask the Task Agent", layout="wide")
st.title("🤖 Ask the Task Agent")

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

# Fetch chat history from FastAPI
API_URL = "http://localhost:8000"  # Change if running elsewhere
def fetch_history(conversation_id):
    resp = requests.get(f"{API_URL}/chat_turns/{conversation_id}")
    return resp.json() if resp.ok else []

history = fetch_history(st.session_state.conversation_id)

# Display chat history
for turn in history:
    st.chat_message("user").write(turn["user_message"])
    st.chat_message("assistant").write(turn["assistant_message"])

# Show example queries to help users
st.markdown("### 💡 **Tips for Better Results**")
st.markdown("""
**📍 Include topic names in your questions for best results!**

**Good examples:**
- "Send a follow-up email to the person who is responsible for **Capstone Project**"
- "How many tasks are related to **Customer Analytics**?"
- "What are the deadlines for **Course Evaluations**?"
- "Show me tasks about **Work Session**"

**Why topic names matter:** Our system uses topic-centered search for
maximum accuracy. Including the specific topic name helps find all related
tasks, people, and deadlines.
""")

# Get new user input
user_query = st.chat_input("Ask about your tasks, people, or topics...")
if user_query:
    st.chat_message("user").write(user_query)
    # Call LangGraph pipeline with conversation_id
    result = run_agent_chat_round(user_query, conversation_id=st.session_state.conversation_id)
    answer = result.get("final_answer", "⚠️ No answer returned.")
    observation = result.get("observation")
    st.chat_message("assistant").write(answer)
    # Store the new turn in FastAPI
    requests.post(f"{API_URL}/chat_turns/", json={
        "conversation_id": st.session_state.conversation_id,
        "user_message": user_query,
        "assistant_message": answer,
        "state": {}  # Optionally store any state
    })
    # Optionally rerun to refresh chat
    st.rerun()

    # Visualization and analysis (unchanged)
    with st.expander("🔍 Show Query-Focused Graph Visualization"):
        try:
            from utils.graphrag import GraphRAG
            rag = GraphRAG()
            if rag.load_graph_with_embeddings():
                graphrag_result = rag.query_with_semantic_reasoning(user_query)
                html_content = rag.generate_visualization_html(user_query, graphrag_result)
                if html_content and not html_content.startswith("<p>Error") and not html_content.startswith("<p>No"):
                    st.markdown("### 🔍 Query Analysis Visualization")
                    st.markdown(f"**Query:** {user_query}")
                    st.markdown(f"**Confidence:** {graphrag_result.get('confidence_score', 0):.3f}")
                    st.components.v1.html(html_content, height=600, scrolling=True)
                    tasks_count = 0
                    people_count = 0
                    dates_count = 0
                    try:
                        import pickle
                        with open("/tmp/topic_graph.gpickle", "rb") as f:
                            graph = pickle.load(f)
                        for node in graphrag_result.get('all_nodes', []):
                            if node in graph:
                                attrs = graph.nodes[node]
                                label = attrs.get('label', '')
                                if label == 'Task':
                                    tasks_count += 1
                                elif label == 'Person':
                                    people_count += 1
                                elif label == 'Date':
                                    dates_count += 1
                    except Exception:
                        evidence = graphrag_result.get('evidence', {})
                        tasks_count = len(evidence.get('tasks', []))
                        people_count = len(evidence.get('people', []))
                        dates_count = len(evidence.get('deadlines', []))
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📋 Tasks Found", tasks_count)
                    with col2:
                        st.metric("👥 People Found", people_count)
                    with col3:
                        st.metric("📅 Deadlines Found", dates_count)
                else:
                    st.warning("📊 Visualization temporarily unavailable")
                    st.info(f"🔧 Debug: html_content length={len(html_content) if html_content else 0}")
                    st.markdown("#### 📝 Query Results (Text)")
                    st.markdown(observation)
            else:
                st.warning("Graph not loaded. Please process emails first.")
        except Exception as e:
            st.error(f"Visualization error: {e}")
            st.markdown("#### 📝 Text Context (Fallback)")
            st.text(observation)