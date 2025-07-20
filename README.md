
# Automated Task Manager

**A production-grade, graph-aware AI assistant for task management, recommendations, and trustworthy GPT answers—grounded in user-uploaded email data and powered by a persistent Neo4j knowledge graph.**

---

## 🚀 Features

- **📧 Email-to-Graph Pipeline**: Upload Gmail Takeout `.mbox` files, extract actionable tasks, and build a rich Neo4j knowledge graph
- **🧠 AI-Powered Extraction**: Modular LangGraph DAGs for robust GPT-based extraction, validation (with HITL), and conversational Q&A
- **🗄️ Neo4j Vector Indexing**: Fast, scalable semantic search and topic matching using native Neo4j vector indexes
- **💬 Intelligent AI Chatbot**: Advanced ChainQA system for natural language queries with conversation memory
- **📅 Visual Task Management**: Interactive calendar view and task organization with Notion integration
- **🔧 Database Management**: Separate controls for Neo4j and Notion database clearing and management
- **👥 Human-in-the-Loop (HITL)**: Pause/resume pipeline for user correction of extractions
- **🚀 Production Ready**: CI/CD pipeline, Docker deployment, and enterprise-grade reliability

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Gmail Takeout (.mbox)] --> B[Preprocessing/Filtering]
    B --> C[LangGraph Extraction Pipeline]
    C --> D[HITL Validation]
    D --> E[Neo4j Graph Construction]
    E --> F[Vector Indexing]
    F --> G[ChainQA AI Chatbot]
    G --> H[Streamlit UI]
    H --> I[Notion Integration]
```

**Graph Schema:**

    (Topic)-[:HAS_TASK]->(Task)-[:RESPONSIBLE_TO]->(Person)-[:HAS_ROLE]->(Role)-[:BELONGS_TO]->(Department)-[:IS_IN]->(Organization)
                    └─[:BASED_ON]->(Summary)
                    └─[:DUE_ON]->(Date)
                    └─[:LINKED_TO]->(Email)

---

## 🔄 ETL Pipeline

**Extract:**
- Gmail Takeout `.mbox` files (no size limits)
- Smart filtering (date, keywords, content length)
- Structured metadata extraction (From, To, Subject, Body, Message-ID)

**Transform:**
- RAG-enhanced extraction (contextual LLM prompt with similar emails)
- LangGraph pipeline: GPT-4 converts unstructured content to structured JSON
- Entity extraction: tasks, people, roles, organizations, dates
- HITL validation for data quality
- Graph construction (Neo4j Cypher)
- Vector embeddings for semantic search

**Load:**
- Neo4j graph database (persistent, scalable)
- Notion database integration for task management
- Multiple interfaces: tabular, calendar, graph query

**Query & Analytics:**
- Conversational AI chatbot with ChainQA reasoning
- Interactive graph visualization
- Natural language task queries

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application with database management |
| `gmail_app.py` | Gmail integration with real-time processing |
| `utils/email_parser.py` | Parses Gmail Takeout `.mbox` into structured DataFrame |
| `utils/embedding.py` | Chunks, embeds, builds semantic index |
| `utils/graph_writer.py` | Converts extracted JSON to Neo4j graph via Cypher |
| `utils/langchain_neo4j_agent.py` | LangChain Neo4jGraph agent for advanced Q&A |
| `utils/chainqa_graph_agent.py` | Multi-step reasoning for complex graph queries |
| `utils/langgraph_unified_memory_agent.py` | Conversation memory and context management |
| `utils/prompt_template.py` | Centralized prompt templates for LLM interactions |
| `utils/langgraph_nodes.py` | LangGraph node definitions for pipeline steps |
| `utils/langgraph_dag.py` | Defines LangGraph DAGs: extraction and chat agent |
| `utils/notion_utils.py` | Notion API integration for task synchronization |
| `pages/My_Calendar.py` | Calendar view of tasks with Notion integration |
| `pages/AI_Chatbot.py` | Enhanced chatbot interface with conversation memory |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container configuration for deployment |
| `docker-compose.yml` | Multi-service deployment with Neo4j |

---

## ⚙️ Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/graph-ai-task-manager.git
   cd graph-ai-task-manager
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or using uv (recommended):
   uv sync
   ```

3. **Set environment variables:** Create `.env` with:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   OPENAI_API_KEY=your_openai_api_key
   NOTION_API_KEY=your_notion_api_key
   NOTION_DATABASE_ID=your_notion_database_id
   ```

4. **Start with Docker Compose (recommended):**
   ```bash
   docker-compose up -d
   ```

5. **Or start manually:**
   ```bash
   streamlit run app.py
   ```

6. **Access the app:**
   - 📱 Streamlit UI: http://localhost:8501
   - 📊 Neo4j Browser: http://localhost:7474
   - 🤖 AI Chatbot: http://localhost:8501/AI_Chatbot
   - 📅 Calendar View: http://localhost:8501/My_Calendar

---

## 🎯 Key Capabilities

### **AI Chatbot Features:**
- **Dynamic Query Processing**: Intelligent routing for task queries, conversation history, and help requests
- **Conversation Memory**: Remembers previous questions and context
- **ChainQA Reasoning**: Multi-step reasoning for complex graph queries
- **Real-time Database Access**: Direct Neo4j queries for accurate task information

### **Task Management:**
- **No Email Limits**: Process entire email archives without artificial size restrictions
- **Human-in-the-Loop Editing**: Review and edit extracted tasks before saving
- **Notion Integration**: Sync tasks with Notion database for external management
- **Database Management**: Separate controls for Neo4j and Notion database clearing

### **Production Features:**
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **Docker Support**: Containerized deployment for scalability
- **Error Handling**: Robust error handling and user feedback
- **Performance Optimized**: Efficient processing and memory management

---

## 🧠 What This Enables

| Feature                        | Enabled by                |
|--------------------------------|---------------------------|
| Structured email → graph       | LangGraph + ChainQA       |
| Persistent knowledge graph     | Neo4j + Vector Indexing   |
| Smart queries                  | LangChain + Cypher        |
| Conversational QA              | ChainQA + Memory Agent    |
| Visual calendar                | Streamlit + Notion        |
| Editable extraction            | HITL validation           |
| Production deployment          | Docker + CI/CD            |

---

## 🚀 Deployment

### **Local Development:**
```bash
streamlit run app.py
```

### **Docker Deployment:**
```bash
docker build -t task-manager .
docker run -p 8501:8501 task-manager
```

### **Cloud Deployment:**
- **Railway**: Connect GitHub repo, set environment variables
- **Render**: Web service with automatic deployments
- **Heroku**: Container deployment with Procfile
- **Google Cloud Run**: Containerized deployment

---

## 📚 References
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [LangChain Neo4jGraph](https://python.langchain.com/docs/integrations/graph/neo4j)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Streamlit](https://streamlit.io/)
- [ChainQA](https://arxiv.org/abs/2304.03442)