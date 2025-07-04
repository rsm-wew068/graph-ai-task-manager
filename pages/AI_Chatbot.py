import sys
import os

# Fix for Hugging Face Spaces: ensure parent directory is in module search path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
from utils.langgraph_dag import run_agent_chat_round

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

# Get new user input
if user_query := st.chat_input("Ask a task-related question..."):
    st.chat_message("user").write(user_query)

    # Run LangGraph reasoning pipeline
    result = run_agent_chat_round(user_query)
    answer = result.get("final_answer", "⚠️ No answer returned.")
    observation = result.get("observation")

    # Show assistant response
    st.chat_message("assistant").write(answer)

    # Show extracted reasoning as expandable
    with st.expander("🔍 Show Graph Reasoning Context"):
        st.markdown("#### Topic Observation")
        st.text(observation)

    # Save to chat history
    st.session_state.chat_history.append({
        "user": user_query,
        "assistant": answer
    })