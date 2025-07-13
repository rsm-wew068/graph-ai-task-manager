# Automated Task Manager: Neo4j + LangGraph + LangChain

A production-grade, graph-aware AI assistant for task management, recommendations, and trustworthy GPT answers—grounded in structured, user-uploaded email data and powered by a persistent Neo4j knowledge graph.

---

## 🚀 Key Features

- **End-to-End Email-to-Graph Pipeline**: Upload Gmail Takeout `.mbox` files, extract actionable tasks, and build a rich knowledge graph in Neo4j.
- **LangGraph Extraction & Reasoning**: Modular DAGs for robust LLM-based extraction, validation (with HITL), and conversational Q&A.
- **Neo4j Vector Indexing**: Fast, scalable semantic search and topic matching using native Neo4j vector indexes.
- **LangChain Agent Integration**: Advanced Cypher generation and answer formatting via LangChain’s Neo4jGraph agent.
- **Streamlit UI**: Intuitive interfaces for upload, validation, calendar view, and conversational chat with graph visualization.
- **Human-in-the-Loop (HITL) Validation**: Pause/resume pipeline for user correction of extractions.

---

## 🏗️ Architecture Overview

```
Gmail Takeout (.mbox) → Preprocessing/Filtering → LangGraph Extraction Pipeline →
HITL Validation → Neo4j Graph Construction → Vector Indexing →
LangChain Agent Q&A → Streamlit UI (Chatbot, Calendar, Visualization)
```

### **Graph Schema**
```
(Topic)-[:HAS_TASK]->(Task)-[:RESPONSIBLE_TO]->(Person)-[:HAS_ROLE]->(Role)-[:BELONGS_TO]->(Department)-[:IS_IN]->(Organization)
                └─[:BASED_ON]->(Summary)
                └─[:DUE_ON]->(Date)
                └─[:LINKED_TO]->(Email)
```

---

## 🔄 ETL Pipeline Overview

This application implements a robust **ETL (Extract, Transform, Load)** pipeline for email-based task management:

### **Extract** 📥
- **Source**: Gmail Takeout `.mbox` files
- **Smart Filtering**: Date range, keywords, content length, email types
- **Email Parsing**: Structured extraction of metadata (From, To, Subject, Body, Message-ID)
- **Content Processing**: First 200MB processed for optimal performance

### **Transform** ⚙️
- **RAG-Enhanced Extraction**: For each email, the system retrieves similar email chunks from your dataset using a local vector index (e.g., FAISS or in-memory embeddings). The LLM prompt is constructed with the main email and up to two related email chunks as additional context. The LLM then extracts structured entities (tasks, people, dates, etc.) from the email, grounded in both the email and its context.
- **LLM Processing**: LangGraph pipeline uses GPT-4 to convert unstructured email content into structured JSON
- **Entity Extraction**: Tasks, people, roles, departments, organizations, dates
- **Data Validation**: Human-in-the-Loop (HITL) validation for data quality
- **Graph Construction**: Hierarchical node formatting with relationship mapping (all via Neo4j Cypher)
- **Semantic Enrichment**: Vector embeddings for similarity search and topic matching (Neo4j vector index)

### **Load** 📊
- **Neo4j Graph Database**: Master knowledge graph with enhanced node hierarchy, persistent and scalable
- **Multiple Interfaces**: 
  - **Tabular View**: Flattened task data in Main Page
  - **Calendar View**: Time-based visualization
  - **Graph Database**: Interactive GraphRAG query system (Neo4j backend)

### **Query & Analytics** 🔍
- **GraphRAG**: Topic-centered semantic search with graph traversal (Neo4j)
- **Natural Language Interface**: Conversational AI chatbot (LangChain agent)
- **Interactive Visualization**: Pyvis-based interactive graph visualization with drag-and-drop, zoom, and filtering capabilities

This ETL approach ensures reliable, scalable processing of large email datasets while maintaining data quality through validation workflows and persistent graph storage in Neo4j.

---

## 🧠 End-to-End Pipeline

1. **User Uploads .mbox Email File**
   - Source: Gmail Takeout
   - Format: `.mbox` file
2. **Preprocessing / Filtering**
   - Extracts metadata (Message-ID, Subject, Date, From, To, CC, Body)
   - Filters out auto-notifications, empty/short replies, out-of-range dates
3. **LangGraph Extraction Pipeline**
   - Nodes: chunk embedding, context retrieval, prompt generation, LLM JSON extraction, validation, HITL, write to Neo4j
   - Pauses for user correction if extraction is invalid
4. **Neo4j Knowledge Graph Storage**
   - All entities and relationships stored in Neo4j with rich schema
   - Embeddings stored as vector properties for semantic search
5. **Neo4j Vector Indexing**
   - Fast topic and node similarity search using native vector index
6. **LangChain Agent Q&A**
   - Natural language queries answered via LangChain agent with Cypher generation and answer formatting
   - Supports advanced, multi-hop graph queries
7. **Streamlit UI**
   - Main Page: Upload, process, validate, view tasks
   - Calendar View: Visualize tasks by due date
   - AI Chatbot: Conversational Q&A with interactive Pyvis graph visualization and query tips
   - HITL Modal: Edit/approve invalid extractions

---

## 🧩 Project Structure

| File | Purpose |
|------|---------|
| `utils/email_parser.py` | Parses Gmail Takeout `.mbox` into structured DataFrame |
| `utils/embedding.py` | Chunks, embeds, builds semantic index |
| `utils/graph_writer.py` | Converts extracted JSON to Neo4j graph via Cypher |
| `utils/graphrag.py` | GraphRAG query system, semantic search (Neo4j vector index) |
| `utils/langchain_neo4j_agent.py` | LangChain Neo4jGraph agent for advanced Q&A |
| `utils/prompt_template.py` | Prompt templates for LLM extraction and reasoning |
| `utils/langgraph_nodes.py` | LangGraph node definitions for pipeline steps |
| `utils/langgraph_dag.py` | Defines LangGraph DAGs: extraction and chat agent |
| `app.py` | Main Streamlit application (home page) |
| `start_services.py` | Service orchestrator (starts both FastAPI and Streamlit) |
| `pages/My_Calendar.py` | Calendar view of tasks |
| `pages/AI_Chatbot.py` | Chatbot interface for graph-based QA |
| `requirements.txt` | Python dependencies |

---

## 🔍 Human-in-the-Loop (HITL) Validation

- Pipeline pauses for user review if:
  - JSON parsing fails
  - JSON is missing required fields
  - No JSON extracted from email
  - Email metadata is inconsistent/incomplete
- User can edit/correct extraction and resume pipeline

---

## 🧠 Conversational Q&A (LangChain Agent)

- User enters a natural language query (e.g., “What tasks are due next week?”)
- LangChain agent:
  1. Performs semantic search (Neo4j vector index)
  2. Generates Cypher queries for Neo4j
  3. Formats answers using LLM
  4. Streams results to UI
- Supports multi-hop, context-rich queries and answer formatting

---

## 🕸️ Neo4j Vector Indexing

- Node embeddings stored as `vector` property on each node
- Vector index created for fast similarity search
- Semantic search and topic matching performed natively in Neo4j

---

## 🖼️ Streamlit UI

- **Main Page**: Upload, process, validate, view tasks
- **Calendar View**: Visualize tasks by due date
- **AI Chatbot**: Conversational Q&A, graph visualization, query tips
- **HITL Modal**: Edit/approve invalid extractions

---

## ⚙️ Configuration & Deployment

- **Neo4j Connection**: Store credentials in `.env` (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`)
- **OpenAI API Key**: Required for LLM processing (`OPENAI_API_KEY`)
- **Docker Support**: Use `host.docker.internal` for cross-container access
- **CI/CD**: GitHub Actions for build, test, and deployment (see `.github/workflows/ci.yml`)
- **Secrets**: Store sensitive values as GitHub Actions secrets

### **Quick Start (GitHub)**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/graph-ai-task-manager.git
   cd graph-ai-task-manager
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set environment variables**: Create `.env` file with Neo4j and OpenAI credentials

4. **Start with Docker Compose** (recommended):
   ```bash
   docker-compose up -d
   ```

5. **Or start manually**:
   ```bash
   python start_services.py
   ```

6. **Access the application**:
   - Streamlit UI: http://localhost:8501
   - FastAPI Backend: http://localhost:8000
   - Neo4j Browser: http://localhost:7474

### **Docker Deployment**

```bash
docker build -t task-manager .
docker run -p 8501:8501 -p 8000:8000 task-manager
```

---

## 📝 Example Usage: LangChain Neo4j Agent

```python
from utils.langchain_neo4j_agent import answer_with_neo4j_agent

query = "Who is responsible for Capstone Project tasks due next week?"
answer = answer_with_neo4j_agent(query)
print(answer)
```

---

## 🧠 What This Enables

| Feature                        | Enabled by                |
|--------------------------------|---------------------------|
| Structured email → graph       | LangGraph                 |
| Persistent knowledge graph     | Neo4j                     |
| Smart queries                  | LangChain + Cypher        |
| Conversational QA              | LangChain agent           |
| Visual calendar                | Streamlit                 |
| Editable extraction            | HITL validation (LangGraph)|

---

## 📚 References
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [LangChain Neo4jGraph](https://python.langchain.com/docs/integrations/graph/neo4j)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Streamlit](https://streamlit.io/)

---