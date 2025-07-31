# Automated Task Manager & Graph Intelligence Platform

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
- PostgreSQL database (Neon recommended)
- Neo4j database (Neo4j AuraDB recommended)
- OpenAI API key

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
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
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

### **📊 Complete LangGraph Architecture with LangSmith Observability**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ORCHESTRATION + LANGSMITH                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Data Loading  │───▶│  Email Filtering│───▶│  RAG Processing  │         │
│  │   Workflow      │    │   Workflow      │    │   Workflow       │         │
│  │   (LangGraph)   │    │   (LangGraph)   │    │   (LangGraph)   │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Task          │───▶│  Validation     │───▶│  Storage        │         │
│  │   Extraction    │    │   Workflow      │    │   Workflow      │         │
│  │   Workflow      │    │   (LangGraph)   │    │   (LangGraph)   │         │
│  │   (LangGraph)   │    └─────────────────┘    └─────────────────┘         │
│  └─────────────────┘              │                       │                 │
│           │                       ▼                       ▼                 │
│           ▼              ┌─────────────────┐    ┌─────────────────┐         │
│  ┌─────────────────┐     │  GraphRAG       │    │  Calendar       │         │
│  │   Chatbot       │◀────│  Workflow       │    │  Workflow       │         │
│  │   Workflow      │     │  (LangGraph)    │    │   (LangGraph)   │         │
│  │   (LangGraph)   │     └─────────────────┘    └─────────────────┘         │
│  └─────────────────┘              │                       │                 │
│           │                       ▼                       ▼                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   ▼                                         │
│                        ┌─────────────────┐                                 │
│                        │  Evaluation     │                                 │
│                        │  Workflow       │                                 │
│                        │  (LangGraph)    │                                 │
│                        └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LANGSMITH OBSERVABILITY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Trace         │    │   Debug         │    │   Monitor       │         │
│  │   Collection    │    │   Interface     │    │   Dashboard     │         │
│  │   (All Nodes)   │    │   (Real-time)   │    │   (Metrics)     │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Performance   │    │   Error         │    │   Quality       │         │
│  │   Analytics     │    │   Tracking      │    │   Assessment    │         │
│  │   (Latency)     │    │   (Debugging)   │    │   (RAGAS)       │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **🔧 Complete LangGraph Workflow System**

#### **1. Data Loading & Filtering Workflow**
```python
# Email loading and filtering pipeline
run_data_loading_workflow(filter_params, database_connection)
```

**Nodes:**
- **`validate_filter_params_node`**: Validates user filter inputs
- **`build_query_node`**: Constructs optimized SQL queries
- **`execute_query_node`**: Runs database queries with error handling
- **`process_results_node`**: Formats and validates loaded data
- **`create_faiss_index_node`**: Builds similarity search index
- **`prepare_chunks_node`**: Creates email chunks for RAG processing

#### **2. Task Extraction Workflow**
```python
# Main extraction workflow
run_extraction_pipeline(email_row, faiss_index, all_chunks, email_index)
```

**Nodes:**
- **`build_prompt_node`**: Creates RAG-enhanced prompts with email context
- **`extract_json_node`**: LLM extracts tasks using structured prompts
- **`attempt_json_parse_node`**: Validates and parses JSON output
- **`pause_for_user_review_node`**: Human-in-the-Loop validation point
- **`process_user_correction_node`**: Handles user corrections
- **`write_graph_node`**: Stores tasks in PostgreSQL and Neo4j

#### **3. Extraction-Only Workflow**
```python
# Pipeline without storage (for preview)
run_extraction_only_pipeline(email_row, faiss_index, all_chunks, email_index)
```

**Nodes:**
- **`build_prompt_node`**: Same RAG prompt creation
- **`extract_json_node`**: LLM extraction
- **`attempt_json_parse_node`**: JSON validation
- **`pause_for_user_review_node`**: HITL validation
- **`process_user_correction_node`**: User corrections
- **`finish_node`**: Returns validated tasks without storage

#### **4. Validation & Quality Control Workflow**
```python
# Human-in-the-loop validation pipeline
run_validation_workflow(extracted_tasks, validation_rules)
```

**Nodes:**
- **`quality_check_node`**: Automated quality assessment
- **`flag_issues_node`**: Identifies tasks needing review
- **`present_for_review_node`**: Shows tasks to user
- **`process_corrections_node`**: Handles user edits
- **`finalize_validation_node`**: Confirms validation completion

#### **5. Storage & Graph Creation Workflow**
```python
# Database and graph storage pipeline
run_storage_workflow(validated_tasks, database_config)
```

**Nodes:**
- **`prepare_postgresql_node`**: Formats data for PostgreSQL
- **`store_tasks_node`**: Inserts tasks into database
- **`prepare_neo4j_node`**: Formats data for Neo4j graph
- **`create_nodes_node`**: Creates graph nodes (Task, Person, Topic, Date)
- **`create_relationships_node`**: Establishes graph relationships
- **`verify_storage_node`**: Confirms successful storage

#### **6. GraphRAG Query Workflow**
```python
# AI chatbot for querying the graph
run_agent_chat_round(user_query)
```

**Nodes:**
- **`classify_query_node`**: Determines query type and entities
- **`search_graph_node`**: Semantic search in Neo4j
- **`expand_context_node`**: Bi-directional graph expansion
- **`generate_answer_node`**: Creates graph-grounded answers
- **`evaluate_answer_node`**: RAGAS evaluation
- **`visualize_graph_node`**: Creates interactive visualizations

#### **7. Calendar & Visualization Workflow**
```python
# Calendar view and task visualization pipeline
run_calendar_workflow(date_range, task_filters)
```

**Nodes:**
- **`load_tasks_node`**: Retrieves tasks from database
- **`filter_by_date_node`**: Applies date filters
- **`format_calendar_data_node`**: Prepares calendar events
- **`create_visualizations_node`**: Generates task charts
- **`prepare_task_details_node`**: Formats task information

#### **8. Evaluation & Analytics Workflow**
```python
# Performance and quality evaluation pipeline
run_evaluation_workflow(extraction_results, user_feedback)
```

**Nodes:**
- **`calculate_metrics_node`**: Computes extraction accuracy
- **`assess_quality_node`**: Evaluates task quality
- **`analyze_patterns_node`**: Identifies improvement areas
- **`generate_report_node`**: Creates evaluation reports
- **`update_models_node`**: Suggests model improvements

### **🎯 Complete User Journey: End-to-End LangGraph Experience**

#### **Phase 1: Data Loading & Filtering (LangGraph Workflow)**
```
1. User clicks "Load Filtered Emails"
   ├── validate_filter_params_node: Validates user inputs
   ├── build_query_node: Constructs optimized SQL
   ├── execute_query_node: Runs database query
   ├── process_results_node: Formats loaded data
   ├── create_faiss_index_node: Builds similarity index
   └── prepare_chunks_node: Creates RAG-ready chunks

2. System confirms data loading success
   ├── Shows loaded email count
   ├── Displays date range and filters applied
   └── Prepares for extraction workflow
```

#### **Phase 2: Task Extraction (LangGraph Workflow)**
```
3. User clicks "Start LLM Processing"
   ├── For each email:
   │   ├── build_prompt_node: Creates RAG-enhanced prompts
   │   ├── extract_json_node: LLM extracts tasks
   │   ├── attempt_json_parse_node: Validates JSON structure
   │   └── If validation fails → pause_for_user_review_node
   └── Returns extraction results with quality metrics

4. Human-in-the-Loop Validation (LangGraph Workflow)
   ├── quality_check_node: Automated quality assessment
   ├── flag_issues_node: Identifies tasks needing review
   ├── present_for_review_node: Shows tasks to user
   ├── User edits task details in interface
   ├── process_corrections_node: Handles user edits
   └── finalize_validation_node: Confirms completion
```

#### **Phase 3: Storage & Graph Creation (LangGraph Workflow)**
```
5. User clicks "Store to Databases"
   ├── prepare_postgresql_node: Formats data for PostgreSQL
   ├── store_tasks_node: Inserts tasks into database
   ├── prepare_neo4j_node: Formats data for Neo4j
   ├── create_nodes_node: Creates graph nodes
   ├── create_relationships_node: Establishes relationships
   └── verify_storage_node: Confirms successful storage
```

#### **Phase 4: Data Exploration & Querying (LangGraph Workflows)**
```
6. User navigates to Calendar View (LangGraph Workflow)
   ├── load_tasks_node: Retrieves tasks from database
   ├── filter_by_date_node: Applies date filters
   ├── format_calendar_data_node: Prepares calendar events
   ├── create_visualizations_node: Generates charts
   └── prepare_task_details_node: Formats task information

7. User asks questions via AI Chatbot (LangGraph Workflow)
   ├── classify_query_node: Understands query intent
   ├── search_graph_node: Semantic search in Neo4j
   ├── expand_context_node: Bi-directional graph expansion
   ├── generate_answer_node: Creates grounded answers
   ├── evaluate_answer_node: RAGAS evaluation
   └── visualize_graph_node: Shows interactive graph
```

#### **Phase 5: Evaluation & Analytics (LangGraph Workflow)**
```
8. System continuously evaluates performance
   ├── calculate_metrics_node: Computes extraction accuracy
   ├── assess_quality_node: Evaluates task quality
   ├── analyze_patterns_node: Identifies improvements
   ├── generate_report_node: Creates evaluation reports
   └── update_models_node: Suggests model improvements
```

### **🔄 LangGraph State Management & Flow Control**

#### **Global Application State**
```python
class ApplicationState(TypedDict, total=False):
    # Data loading state
    loaded_emails: List[Dict]
    faiss_index: Any
    email_chunks: List[str]
    filter_params: Dict
    
    # Extraction state
    extraction_results: List[Dict]
    validation_queue: List[Dict]
    quality_metrics: Dict
    
    # Storage state
    stored_tasks: List[Dict]
    graph_nodes: List[Dict]
    graph_relationships: List[Dict]
    
    # UI state
    current_page: str
    user_preferences: Dict
    session_data: Dict
```

#### **Data Loading State**
```python
class DataLoadingState(TypedDict, total=False):
    filter_params: Dict
    sql_query: str
    query_results: List[Dict]
    processed_data: List[Dict]
    faiss_index: Any
    email_chunks: List[str]
    loading_status: str
    error_message: str
```

#### **Extraction State**
```python
class ExtractionState(TypedDict, total=False):
    current_email_row: Dict
    email_text: str
    faiss_index: Any
    all_chunks: List[str]
    email_index: str
    retry_count: int
    extracted_json: str
    validated_json: Dict
    valid: bool
    needs_user_review: bool
    error: str
    graph_stored: bool
    database_stored: bool
    quality_score: float
    extraction_confidence: float
```

#### **Validation State**
```python
class ValidationState(TypedDict, total=False):
    tasks_to_review: List[Dict]
    validation_rules: Dict
    quality_metrics: Dict
    flagged_issues: List[Dict]
    user_corrections: List[Dict]
    validation_status: str
    review_complete: bool
```

#### **Storage State**
```python
class StorageState(TypedDict, total=False):
    tasks_to_store: List[Dict]
    postgresql_ready: bool
    neo4j_ready: bool
    storage_progress: Dict
    stored_count: int
    error_count: int
    storage_complete: bool
```

#### **Chat State**
```python
class ChatState(TypedDict, total=False):
    user_query: str
    query_type: str
    entities: List[str]
    search_results: List[Dict]
    expanded_context: List[Dict]
    final_answer: str
    ragas_score: float
    graph_visualization: str
    conversation_history: List[Dict]
    query_confidence: float
```

#### **Calendar State**
```python
class CalendarState(TypedDict, total=False):
    date_range: Dict
    task_filters: Dict
    calendar_events: List[Dict]
    task_details: Dict
    visualization_data: Dict
    navigation_state: Dict
```

#### **Evaluation State**
```python
class EvaluationState(TypedDict, total=False):
    extraction_metrics: Dict
    quality_scores: Dict
    user_feedback: List[Dict]
    performance_data: Dict
    improvement_suggestions: List[str]
    evaluation_report: Dict
```

### **🎛️ Human-in-the-Loop Integration**

#### **Validation Triggers**
- **JSON parsing errors**: Invalid structure or missing fields
- **Low confidence scores**: LLM uncertainty in extraction
- **Missing required fields**: task_name, message_id, etc.
- **User-initiated review**: Manual validation requests

#### **Correction Flow**
```
1. System pauses at pause_for_user_review_node
2. Streamlit shows correction interface
3. User edits JSON in text area
4. User clicks "Validate & Continue"
5. process_user_correction_node processes changes
6. Pipeline continues to storage
```

### **📊 Error Handling & Resilience**

#### **Pipeline Resilience**
- **Retry logic**: Failed extractions retry up to 3 times
- **Graceful degradation**: Continues processing even if some emails fail
- **Error logging**: Detailed error tracking for debugging
- **Fallback mechanisms**: Alternative processing paths

#### **Data Validation**
- **Schema validation**: Ensures flat JSON structure
- **Date validation**: Only valid dates stored in Neo4j
- **Relationship validation**: Proper entity linking
- **Content validation**: Meaningful task descriptions

### **🚀 Performance Optimizations**

#### **Batch Processing**
- **Email batching**: Process emails in configurable batches
- **Parallel processing**: Concurrent email processing
- **Memory management**: Efficient FAISS index usage
- **Database optimization**: Bulk inserts and transactions

#### **Caching & Persistence**
- **Session state**: Maintains data across page navigation
- **Database persistence**: Reliable storage in PostgreSQL/Neo4j
- **Graph caching**: Efficient Neo4j query patterns
- **Embedding caching**: Reuse FAISS indices

### **🔍 LangSmith Observability & Monitoring**

#### **Trace Collection & Debugging**
- **Complete workflow tracing**: Every LangGraph node execution is traced
- **Real-time debugging**: Live inspection of node inputs/outputs
- **Error tracking**: Automatic capture and analysis of failures
- **Performance profiling**: Latency and throughput metrics for each node

#### **Monitoring & Analytics**
- **Live dashboard**: Real-time monitoring of all workflows
- **Performance metrics**: Response times, success rates, error rates
- **Quality assessment**: Integration with RAGAS evaluation metrics
- **Resource utilization**: Memory, CPU, and API usage tracking

#### **Development & Production Benefits**
- **Development**: Debug complex workflows step-by-step
- **Testing**: Validate workflow behavior with trace analysis
- **Production**: Monitor system health and performance
- **Optimization**: Identify bottlenecks and improvement opportunities

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

---

**Built with ❤️ using LangGraph, Neo4j, and Streamlit**