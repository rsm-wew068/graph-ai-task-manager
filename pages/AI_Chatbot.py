import streamlit as st
from utils.langgraph_dag import run_agent_chat_round

st.set_page_config(page_title="🤖 Ask the Task Agent", layout="wide")
st.title("🤖 Ask the Task Agent")

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