import streamlit as st
from utils.langgraph_dag import run_extraction_pipeline
from utils.email_parser import parse_uploaded_file
from utils.embedding import embed_dataframe
import pandas as pd

st.set_page_config(page_title="Automated Task Manager", layout="wide")
st.title("📬 Automated Task Manager")

st.markdown("""
Upload your email archive to extract structured tasks.  
Then use the sidebar to:
- 🗓 View extracted tasks in a calendar format
- 🧠 Ask topic-related questions

This app uses LangGraph + GPT + graph reasoning to help you manage unstructured email chaos.
""")

uploaded_file = st.file_uploader("Upload your Gmail Takeout ZIP file (containing .mbox)", type=["zip"])

if uploaded_file is not None:
    with st.spinner("Processing your file..."):
        df_emails = parse_uploaded_file(uploaded_file)
        index, all_chunks = embed_dataframe(df_emails)

        outputs = []
        for email_text in df_emails["content"].tolist():
            result = run_extraction_pipeline(email_text, index, all_chunks)
            outputs.append(result)

        st.success("Extraction complete!")
        graphs = [res["graph"] for res in outputs if "graph" in res]
        tasks = [res["validated_json"] for res in outputs if "validated_json" in res]

        if tasks:
            st.subheader("📋 Extracted Tasks")
            st.dataframe(pd.DataFrame(tasks))

        if graphs:
            st.subheader("📊 Task Graph (last processed)")
            st.graphviz_chart(graphs[-1])