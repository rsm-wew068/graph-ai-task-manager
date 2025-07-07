import sys
import os

# Robust path fix for Hugging Face Spaces
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"AI_Chatbot - Parent directory: {parent_dir}")
print(f"Utils directory exists: {os.path.exists(os.path.join(parent_dir, 'utils'))}")

import streamlit as st

try:
    from utils.langgraph_dag import run_agent_chat_round
    print("✅ AI_Chatbot - Successfully imported utils")
except ImportError as e:
    print(f"❌ AI_Chatbot - Import error: {e}")
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(page_title="🤖 Ask the Task Agent", layout="wide")
st.title("🤖 Ask the Task Agent")

# Show data status
if (hasattr(st.session_state, 'processing_complete') and
        st.session_state.processing_complete):
    outputs = st.session_state.get("extracted_tasks", [])
    valid_tasks_count = len([
        res for res in outputs
        if "validated_json" in res and res.get("valid", False)
    ])
    st.success(
        f"📊 Ready to answer questions about {valid_tasks_count} tasks "
        f"from your processed emails"
    )
elif (hasattr(st.session_state, 'parsing_complete') and
      st.session_state.parsing_complete):
    st.warning(
        "📁 Emails parsed but not yet processed with LLM. "
        "Go to the main page to start LLM processing for Q&A."
    )
else:
    st.info(
        "📝 No data loaded yet. Please upload and process emails "
        "from the main page first to enable Q&A."
    )

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    st.chat_message("user").write(message["user"])
    st.chat_message("assistant").write(message["assistant"])

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
if user_query := st.chat_input("Ask about your tasks, people, or topics..."):
    st.chat_message("user").write(user_query)

    # Run LangGraph reasoning pipeline
    result = run_agent_chat_round(user_query)
    answer = result.get("final_answer", "⚠️ No answer returned.")
    observation = result.get("observation")

    # Show assistant response
    st.chat_message("assistant").write(answer)

    # Show extracted reasoning as expandable with beautiful visualization
    with st.expander("🔍 Show Query-Focused Graph Visualization"):
        try:
            # Generate GraphRAG visualization
            from utils.graphrag import GraphRAG
            
            # Create GraphRAG instance and run the query to get result data
            rag = GraphRAG()
            if rag.load_graph_with_embeddings():
                # Re-run the query to get the structured result for visualization
                graphrag_result = rag.query_with_semantic_reasoning(user_query)
                
                # Generate visualization HTML directly (no file I/O)
                html_content = rag.generate_visualization_html(user_query, graphrag_result)
                
                # Display the visualization in Streamlit
                if html_content and not html_content.startswith("<p>Error") and not html_content.startswith("<p>No"):
                    st.markdown("### 🔍 Query Analysis Visualization")
                    st.markdown(f"**Query:** {user_query}")
                    st.markdown(f"**Confidence:** {graphrag_result.get('confidence_score', 0):.3f}")
                    
                    # Show the interactive graph
                    st.components.v1.html(html_content, height=600, scrolling=True)
                    
                    # Show summary stats based on the actual GraphRAG result
                    # Count nodes by type from the graph result
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
                        # Fallback: try to parse from old evidence format
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
                    # Show text fallback
                    st.markdown("#### 📝 Query Results (Text)")
                    st.markdown(observation)
            else:
                st.warning("Graph not loaded. Please process emails first.")
                
        except Exception as e:
            st.error(f"Visualization error: {e}")
            # Fallback to text display
            st.markdown("#### 📝 Text Context (Fallback)")
            st.text(observation)

    # Save to chat history
    st.session_state.chat_history.append({
        "user": user_query,
        "assistant": answer
    })