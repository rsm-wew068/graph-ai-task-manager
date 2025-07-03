---
title: Automated Task Manager
emoji: 🚀
colorFrom: pink
colorTo: pink
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Upload email archives to extract tasks, owners, and deadline
license: mit
---

# Automated Task Manager 🧠📅  
A graph-aware reasoning assistant for task understanding, recommendations, and trustable GPT answers — grounded in structured user-uploaded email data.

---

## 📦 Project Structure

The project is organized into modular components for graph construction, reasoning, LangGraph workflows, and Streamlit UI delivery:

| File | Purpose |
|------|---------|
| `project_graph.gpickle` | Saved graph object |
| `utils/email_parser.py` | Parses Gmail Takeout ZIP into structured email DataFrame |
| `utils/embedding.py` | Chunks, embeds, builds FAISS index |
| `utils/graph_writer.py` | Converts extracted task JSON to NetworkX graph |
| `utils/prompt_template.py` | Prompt templates for GPT-based reasoning and extraction |
| `utils/langgraph_nodes.py` | LangGraph node definitions for each pipeline step |
| `utils/langgraph_dag.py` | Defines DAGs: agent chat and email-to-graph extraction |
| `utils/tools.py` | Graph traversal, visualization, project name inference |
| `app.py` | Streamlit entry: upload ZIP, run extraction pipeline |
| `pages/1_🗓_Calendar_View.py` | Monthly calendar view of extracted tasks |
| `pages/2_🤖_Ask_the_Agent.py` | Chatbot interface for graph-based QA |
| `requirements.txt` | Python dependency list |

---

## 🧠 End-to-End Reasoning Workflow (Agent Chat)

1. **User Input** → User submits a natural language question (e.g., “What’s due for LNG next week?”)
2. **Project Inference** → GPT infers referenced project
3. **Graph Query** → Query relevant project/task/person subgraph
4. **Observation Generation** → Create structured context string
5. **Answer Generation** → GPT returns a trustable answer
6. **UI Display** → Shown in a chatbot with answer + expandable reasoning context

---

## 🧠 Full LangGraph Pipeline (Triggered on ZIP Upload)

1. **User Uploads Gmail ZIP (containing `.mbox`)**
2. **Clean & Normalize** → Parse emails, sanitize content
3. **Embed & Store in FAISS**
4. **Retrieve Chunks (RAG)**
5. **GPT Extraction → Structured JSON**
6. **Validate JSON**  
   - Retry if invalid  
   - Optionally pause for user corrections
7. **Save to Graph** (Project–Task–Person schema)
8. **GraphRAG QA** (triggered later via agent)
9. **UI Display (Agent Chat)** → Final answer + graph + observation
10. **UI Display (Calendar View)** → Monthly task visualization

---

## 🚀 How to Use the App

### Main Page (`app.py`)
- Upload your Gmail Takeout ZIP file
- Extract tasks and build graph memory

### Page 1: 🗓 Calendar View
- View extracted tasks in a calendar grid by due/start date
- Filter by owner or department

### Page 2: 🤖 Ask the Agent
- Live chatbot to ask project-related questions
- Responses grounded in graph context