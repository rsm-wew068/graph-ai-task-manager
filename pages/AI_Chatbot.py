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

# Show example queries to help users (stays at top)
st.markdown("""
### 🤖 Ask Anything About Your Tasks, People, or Deadlines
- "Who is collaborating with Frank Hayden?"
- "What tasks are due next week?"
- "List all tasks assigned to Susan M. Scott."
- "Show me all tasks related to Enron."
- "Who owns the task 'Review document'?"

**Tip:** You can ask about people, deadlines, task names, or any detail in your graph!
""")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    st.chat_message("user").write(message["user"])
    st.chat_message("assistant").write(message["assistant"])

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
            # Generate Neo4j GraphRAG visualization
            from utils.neo4j_graphrag import Neo4jGraphRAG
            
            # Create Neo4j GraphRAG instance and run the query to get result data
            rag = Neo4jGraphRAG()
            if rag.load_graph_with_embeddings():
                # Re-run the query to get the structured result for visualization
                graphrag_result = rag.query_with_semantic_reasoning(user_query)
                
                # Generate visualization HTML directly (no file I/O)
                html_content = rag.generate_visualization_html(user_query, graphrag_result)
                
                # Display the visualization in Streamlit
                if html_content and not html_content.startswith("<p>Error") and not html_content.startswith("<p>No"):
                    st.markdown("### 🔍 Query Analysis Visualization")
                    st.markdown(f"**Query:** {user_query}")
                    
                    # Display RAGAS evaluation if available
                    if graphrag_result.get('ragas_scores'):
                        st.markdown("### 📊 RAGAS Evaluation")
                        
                        # Show RAGAS summary
                        ragas_summary = graphrag_result.get('ragas_summary', '')
                        if ragas_summary:
                            st.markdown(ragas_summary)
                        
                        # Show individual RAGAS metrics in columns
                        ragas_scores = graphrag_result.get('ragas_scores', {})
                        if ragas_scores:
                            cols = st.columns(len(ragas_scores))
                            for i, (metric, score) in enumerate(ragas_scores.items()):
                                metric_display = {
                                    'faithfulness': '🎯 Faithfulness',
                                    'answer_relevancy': '🔍 Relevancy', 
                                    'llm_context_precision_without_reference': '📋 Precision'
                                }.get(metric, metric)
                                
                                cols[i].metric(metric_display, f"{score:.3f}")
                    else:
                        # Fallback to simple confidence
                        st.markdown(f"**Quick Assessment:** {graphrag_result.get('confidence_score', 0):.3f}")
                    
                    # Show the interactive graph
                    st.components.v1.html(html_content, height=600, scrolling=True)
                    
                    # Show summary stats based on the actual Neo4j GraphRAG result
                    # Count nodes by type from the graph result
                    tasks_count = 0
                    people_count = 0
                    dates_count = 0
                    
                    try:
                        # Get node counts from Neo4j GraphRAG result
                        with rag.driver.session() as session:
                            all_nodes = graphrag_result.get('all_nodes', [])
                            if all_nodes:
                                # Query Neo4j to get node types for the found nodes
                                result = session.run("""
                                    UNWIND $node_names as name
                                    MATCH (n {name: name})
                                    RETURN labels(n)[0] as label, count(n) as count
                                """, node_names=all_nodes)
                                
                                for record in result:
                                    label = record["label"]
                                    count = record["count"]
                                    if label == 'Task':
                                        tasks_count += count
                                    elif label == 'Person':
                                        people_count += count
                                    elif label == 'Date':
                                        dates_count += count
                    except Exception:
                        # Fallback: estimate from node names in result
                        all_nodes = graphrag_result.get('all_nodes', [])
                        # Simple heuristic based on typical node name patterns
                        for node in all_nodes:
                            if any(word in node.lower() for word in ['task', 'deliverable', 'action']):
                                tasks_count += 1
                            elif '(' in node and ')' in node:  # Person nodes have format "Name (Role)"
                                people_count += 1
                            elif any(char.isdigit() for char in node):  # Date nodes contain numbers
                                dates_count += 1
                    
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