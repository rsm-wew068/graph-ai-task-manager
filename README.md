# AI Task Management System

Extract, manage, and reason over tasks, people, and knowledge from your emails—using AI, graphs, and explainable automation powered by LangGraph, Neo4j, and RAGAS.

## 🚀 Features

### **Email Processing & Task Extraction**
- **Smart Email Filtering**: Load emails with date ranges, sender filters, and limits
- **AI-Powered Task Extraction**: Extract tasks using LangGraph and GPT-4
- **Human-in-the-Loop Validation**: Review and correct extracted tasks before storage
- **Flat JSON Structure**: Clean, consistent task data format

### **Database Integration**
- **PostgreSQL Storage**: Reliable relational storage for emails and tasks
- **Neo4j Graph Database**: Rich relationship modeling between entities
- **Automatic Data Sync**: Tasks stored in both databases with proper relationships

### **Interactive Interfaces**
- **📅 Calendar View**: Time-based task visualization with smart navigation
- **🤖 AI Chatbot**: Natural language queries about tasks, people, and deadlines
- **📊 Task Management**: Review, validate, and store extracted tasks

### **Graph Intelligence**
- **Entity-Agnostic Queries**: Ask about any entity type (Person, Task, Topic, etc.)
- **Bi-Directional Graph Expansion**: Rich context from all relationship directions
- **Interactive Visualizations**: See task relationships and dependencies
- **RAGAS Evaluation**: Trust scores for every AI-generated answer
- **LangSmith Observability**: Complete workflow tracing and debugging

## 🛠️ Tech Stack

- **Frontend**: Streamlit (interactive web interface)
- **AI/ML**: LangGraph, OpenAI GPT-4, FAISS (vector search)
- **Observability**: LangSmith (monitoring, debugging, tracing)
- **Databases**: PostgreSQL (relational), Neo4j (graph)
- **Evaluation**: RAGAS (answer quality assessment)
- **Data Processing**: Pandas, PyArrow (Parquet support)

## 📦 Installation

### Prerequisites
- Python 3.12+
- PostgreSQL database (Neon)
- Neo4j database (Neo4j AuraDB)
- OpenAI API key

### Optional
- Docker (for local image builds; GitHub Actions Docker stage is temporarily disabled to avoid runner disk-space exhaustion)

### Setup
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd graph-ai-task-manager
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or using uv
   uv sync
   ```

3. **Environment Configuration**
   Create a `.env` file with your credentials:
```env
# Required: OpenAI and Database Configuration
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_postgresql_connection_string
NEO4J_URI=your_neo4j_connection_string
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password

# Optional: LangSmith Configuration (for tracing and debugging)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=graph-ai-task-manager
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```

4. **Database Setup**
   ```bash
   # Create tables in PostgreSQL
   psql -d your_database -f create_tables.sql
   ```

## 🎯 Quick Start

### 1. Load Sample Data
```bash
python create_and_load_sample.py
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 2b. Run with Docker (local)
```bash
docker build -t graph-ai-task-manager:local .
docker run --rm -p 8501:8501 --env-file .env graph-ai-task-manager:local
```
> Note: The GitHub Actions Docker job is currently disabled to conserve runner disk space; build locally when needed.

### 3. Extract Tasks
- Click "Load Filtered Emails" to load emails from database
- Choose "Most recent first" or "Random sample" for historical data
- Click "Start LLM Processing" to extract tasks
- Review and validate tasks using Human-in-the-Loop interface
- Click "Store to Databases" to save tasks

### 4. Explore Your Data
- **📅 Calendar View**: Navigate to different time periods, click tasks for details
- **🤖 AI Chatbot**: Ask natural language questions about your tasks

## 🔄 LangGraph End-to-End Pipeline

**LangGraph Orchestration Workflows:**

1. **Data Loading** - Initial data ingestion and preprocessing
2. **Email Filtering** - Email classification and filtering
3. **RAG Processing** - Retrieval-Augmented Generation processing
4. **Task Extraction** - Task identification and extraction
5. **Validation** - Data validation and quality checks
6. **Storage** - Data persistence and management
7. **GraphRAG** - Graph-based retrieval and reasoning
8. **Calendar** - Calendar integration and scheduling
9. **Chatbot** - Interactive conversation handling
10. **Evaluation** - Performance assessment and metrics

**LangSmith Observability Components:**

1. **Trace Collection** - Comprehensive tracing across all nodes
2. **Debug Interface** - Real-time debugging capabilities
3. **Monitor Dashboard** - Metrics and performance monitoring
4. **Performance Analytics** - Latency and throughput analysis
5. **Error Tracking** - Debugging and error management
6. **Quality Assessment** - RAGAS-based quality evaluation

### **🔧 Key Workflow Components**

**Core Processing Pipelines:**
- **Data Loading**: Email filtering, database queries, FAISS index creation
- **Task Extraction**: RAG-enhanced LLM extraction with JSON validation
- **Human-in-the-Loop**: Quality checks, user corrections, validation
- **Storage**: PostgreSQL and Neo4j dual storage with relationship creation
- **GraphRAG**: Semantic search, context expansion, answer generation
- **Calendar**: Task visualization, date filtering, interactive navigation
- **Evaluation**: Performance metrics, quality assessment, RAGAS scoring

### **🎯 User Journey Overview**

**Typical Workflow:**
1. **Load Emails** → Filter and load emails from database
2. **Extract Tasks** → AI-powered task extraction with validation
3. **Review & Correct** → Human-in-the-loop validation and corrections
4. **Store Data** → Save to PostgreSQL and Neo4j with relationships
5. **Explore** → Use calendar view and AI chatbot for insights
6. **Evaluate** → Continuous performance monitoring and quality assessment

### **🔄 State Management**

**Key State Components:**
- **Data Loading**: Email data, FAISS index, filter parameters
- **Extraction**: Current email processing, validation queue, quality metrics
- **Storage**: Task data, graph nodes, relationships
- **UI**: Current page, user preferences, session data
- **Chat**: Query processing, search results, conversation history
- **Calendar**: Date ranges, task filters, visualization data

### **🎛️ Human-in-the-Loop Integration**

**Validation Triggers:**
- JSON parsing errors, low confidence scores, missing required fields
- User-initiated review requests

**Correction Flow:**
1. System pauses for user review
2. User edits task details in interface
3. Validation and continuation of pipeline

### **📊 Error Handling & Resilience**

**Pipeline Resilience:**
- Retry logic (up to 3 attempts), graceful degradation, error logging
- Fallback mechanisms for alternative processing paths

**Data Validation:**
- Schema validation, date validation, relationship validation, content validation

### **🚀 Performance Optimizations**

**Batch Processing:**
- Email batching, parallel processing, memory management, database optimization

**Caching & Persistence:**
- Session state, database persistence, graph caching, embedding caching

### **🔍 LangSmith Observability & Monitoring**

**Trace Collection & Debugging:**
- Complete workflow tracing, real-time debugging, error tracking, performance profiling

**Monitoring & Analytics:**
- Live dashboard, performance metrics, quality assessment, resource utilization

**Benefits:**
- Development debugging, testing validation, production monitoring, optimization insights

## 📊 Data Structure

### Task Schema (Flat Structure)
```json
{
  "task_name": "Review quarterly report",
  "task_description": "Review and approve Q4 financial report",
  "topic": "Financial Review",
  "status": "not started",
  "priority_level": "High",
  "sender": "John Smith",
  "assigned_to": "Jane Doe",
  "due_date": "2024-12-31",
  "received_date": "2024-12-15",
  "message_id": "<unique-email-id>",
  "spam": false
}
```

### Graph Relationships
- **Task** → `ASSIGNED_TO` → **Person**
- **Task** → `DUE_ON` → **Date**
- **Task** → `LINKED_TO` → **EmailIndex**
- **Topic** → `HAS_TASK` → **Task**
- **Person** → `SENT_TASK` → **Task**

## 🗣️ Example Queries

Ask the AI chatbot about:
- **People**: "Who is collaborating with Frank Hayden?"
- **Deadlines**: "What tasks are due next week?"
- **Assignments**: "List all tasks assigned to Susan M. Scott."
- **Topics**: "Show me all tasks related to Enron."
- **Specific Tasks**: "Who owns the task 'Review document'?"

## 📅 Calendar Navigation

The calendar automatically opens at the earliest task due date and provides:
- **Month/Year Selection**: Quick navigation to any time period
- **Smart Defaults**: Opens at most relevant date for your data
- **Task Details**: Click any task to see full information
- **Historical Data Support**: Optimized for datasets like Enron (1980-2006)

## 🔧 Configuration

### CI/CD Notes
- GitHub Actions runs linting, formatting, and mypy. Docker build is **temporarily disabled** to avoid "No space left on device" on hosted runners. Re-enable in `.github/workflows/ci.yml` when runner space is sufficient.

### Email Filtering Options
- **Date Ranges**: Last 7/30/90 days, this year, custom ranges
- **Sender Filters**: Filter by email domain or sender name
- **Load Limits**: Control how many emails to process (1-100,000)

### Task Extraction Settings
- **Processing Limits**: Control LLM processing batch size
- **Debug Mode**: Detailed extraction logs
- **Validation**: Human-in-the-Loop task review

## 🚨 Troubleshooting

### Common Issues
1. **"No tasks extracted"**: Check email content quality and LLM API key
2. **Database connection errors**: Verify PostgreSQL and Neo4j credentials
3. **Large file uploads**: Use filtering to reduce email count
4. **Calendar not showing tasks**: Ensure tasks have valid dates

### Performance Tips
- Start with 100-1000 emails for testing
- Use "Random sample" for diverse data
- Enable debug mode for detailed logs
- Store tasks in batches for large datasets

## 📈 Roadmap

- [ ] **Advanced Filtering**: Content-based email filtering
- [ ] **Task Templates**: Predefined task extraction patterns
- [ ] **Export Features**: CSV/JSON export of extracted tasks
- [ ] **Team Collaboration**: Multi-user task management
- [ ] **Advanced Analytics**: Task completion trends and insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
