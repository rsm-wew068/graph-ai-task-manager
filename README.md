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
short_description: Extract tasks from Gmail with AI-powered GraphRAG
license: mit
---

# Automated Task Manager 🧠📅  
A graph-aware reasoning assistant for task understanding, recommendations, and trustable GPT answers — grounded in structured user-uploaded email data.

**✨ Recent Improvements**: Enhanced topic matching, improved node hierarchy formatting, comprehensive user guidance for optimal query results, and streamlined codebase with removal of obsolete fallback upload methods.

---

## 📦 Project Structure

The project is organized into modular components for graph construction, reasoning, LangGraph workflows, and Streamlit UI delivery:

| File | Purpose |
|------|---------|
| `topic_graph.gpickle` | Saved graph object |
| `utils/email_parser.py` | Parses Gmail Takeout .mbox into structured email DataFrame |
| `utils/embedding.py` | Chunks, embeds, builds FAISS index |
| `utils/graph_writer.py` | Converts extracted task JSON to NetworkX graph |
| `utils/graphrag.py` | GraphRAG query system for topic-centered search |
| `utils/prompt_template.py` | Prompt templates for GPT-based reasoning and extraction |
| `utils/langgraph_nodes.py` | LangGraph node definitions for each pipeline step |
| `utils/langgraph_dag.py` | Defines DAGs: agent chat and email-to-graph extraction |
| `app.py` | Streamlit entry: upload .mbox, run extraction pipeline |
| `pages/My_Calendar.py` | Monthly calendar view of extracted tasks |
| `pages/AI_Chatbot.py` | Chatbot interface for graph-based QA |
| `requirements.txt` | Python dependency list |

---

## 🧠 End-to-End Reasoning Workflow (AI Chatbot)

1. **User Input** → User submits a natural language question (e.g., "How many tasks are related to Capstone project?")
2. **Topic-Centered Search** → Enhanced topic matching with:
   - **Semantic Similarity**: Uses sentence transformers (threshold 0.5)
   - **Substring Matching**: Catches topic variations (e.g., "Capstone project - Praxis" vs "Capstone project - Praxis (email insights)")
   - **Flexible Query**: Includes all relevant topic matches for comprehensive results
3. **Graph Expansion** → Traverses up to 25 related nodes following meaningful relationships:
   - From **Topics** → **Tasks** via `HAS_TASK`
   - From **Tasks** → **People**, **Dates**, **Summaries** via `RESPONSIBLE_TO`, `COLLABORATED_BY`, `DUE_ON`, etc.
   - From **People** → **Roles** → **Departments** → **Organizations** via hierarchical relationships
4. **Visualization Generation** → Creates interactive HTML graph with hover details
5. **LangGraph Reasoning** → GPT provides intelligent natural language answers
6. **UI Display** → Chatbot shows:
   - **Answer**: Natural language response
   - **Graph Visualization**: Interactive network of related nodes
   - **Query Tips**: Guidance for better results

---

## 🧠 Full LangGraph Pipeline (Triggered on .mbox Upload)

1. **User Uploads Inbox.mbox file** (Gmail Takeout extracted)
2. **Smart Email Filtering** → Apply intelligent filters:
   - **Date Range**: Filter by date (default: last 180 days)
   - **Keywords**: Only parse emails containing specific terms
   - **Content Length**: Skip very short emails
   - **Exclude Types**: Skip notifications, newsletters, automated emails
   - **Safety Limits**: Max 2000 emails, first 200MB processed
3. **Clean & Normalize** → Parse emails, sanitize content
4. **Embed & Store in FAISS** → Create vector embeddings for similarity search
5. **Retrieve Chunks (RAG)** → Find relevant email content for each email
6. **GPT Extraction → Structured JSON** → Extract tasks with complete metadata
7. **Validate JSON (Human-in-the-Loop)**  
   - **If valid**: Continue to step 8
   - **If invalid**: Pause and show user the raw JSON for correction
   - **User options**: Edit JSON, skip email, or stop processing
   - **After correction**: Resume pipeline with validated JSON
8. **Build NetworkX Graph** → Create structured graph with enhanced node formatting:
   - **Person nodes**: `"Name (Role)"` (e.g., "Rachel Wang (Student Team Lead)")
   - **Role nodes**: `"Role (Department)"` (e.g., "Student Team Lead (Praxis Capstone Project)")
   - **Department nodes**: `"Department (Organization)"` (e.g., "Praxis Capstone Project (University of California, San Diego)")
   - **Organization nodes**: Just organization name (e.g., "University of California, San Diego")
9. **Combine Graphs** → Merge individual graphs into master knowledge graph
10. **GraphRAG QA** → Enable topic-centered queries with semantic search
11. **UI Display** → Three interfaces:
    - **Main Page**: Upload, process, validate, view extracted tasks
    - **Calendar View**: Time-based visualization of tasks
    - **AI Chatbot**: Natural language Q&A with graph visualization

### 🔍 Human-in-the-Loop (HITL) Validation

The pipeline pauses for user review when:
- JSON parsing fails (syntax errors)
- JSON is valid but missing expected fields (name, deliverable, task, or Topic structure)
- No JSON was extracted from the email content
- Email metadata is inconsistent or incomplete

**Enhanced Entity Extraction & Graph Construction**:
- **Complete Email Metadata**: GPT receives Message-ID, From, To, Cc, Bcc, names, dates, subject, and content
- **Organization Detection**: Automatically extracts organizations from email domains (@company.com → "Company")
- **Rich People Information**: Captures names, roles, departments from email headers and signatures
- **Accurate References**: Uses actual Message-ID for email_index ensuring data integrity
- **Hierarchical Node Formatting**: 
  - **Person**: `"Name (Role)"` for clear identification
  - **Role**: `"Role (Department)"` maintaining existing format
  - **Department**: `"Department (Organization)"` - solves hierarchy conflicts
  - **Organization**: Clean organization name only
- **Relationship Preservation**: Maintains `RESPONSIBLE_TO` vs `COLLABORATED_BY` edge distinctions
- **Topic Variation Handling**: Semantic similarity captures related topic names automatically

This ensures data quality and allows debugging of prompt failures.

---

## 🔄 **ETL Pipeline Overview**

This application implements a comprehensive **ETL (Extract, Transform, Load)** pipeline for email-based task management:

### **Extract** 📥
- **Source**: Gmail Takeout Inbox.mbox files
- **Smart Filtering**: Date range, keywords, content length, email types
- **Email Parsing**: Structured extraction of metadata (From, To, Subject, Body, Message-ID)
- **Content Processing**: First 200MB processed for optimal performance

### **Transform** ⚙️
- **LLM Processing**: GPT-4 converts unstructured email content into structured JSON
- **Entity Extraction**: Tasks, people, roles, departments, organizations, dates
- **Data Validation**: Human-in-the-Loop (HITL) validation for data quality
- **Graph Construction**: Hierarchical node formatting with relationship mapping
- **Semantic Enrichment**: Vector embeddings for similarity search and topic matching

### **Load** 📊
- **NetworkX Graph**: Master knowledge graph with enhanced node hierarchy
- **Persistent Storage**: Session state management across UI pages
- **Multiple Interfaces**: 
  - **Tabular View**: Flattened task data in Main Page
  - **Calendar View**: Time-based visualization
  - **Graph Database**: Interactive GraphRAG query system

### **Query & Analytics** 🔍
- **GraphRAG**: Topic-centered semantic search with graph traversal
- **Natural Language Interface**: Conversational AI chatbot
- **Interactive Visualization**: HTML-based graph exploration with hover details

This ETL approach ensures reliable, scalable processing of large email datasets while maintaining data quality through validation workflows.

---

## 🏗️ **Technical Architecture**

### **GraphRAG Query System**
The application uses an advanced GraphRAG (Graph Retrieval-Augmented Generation) system for intelligent task querying:

**Query Processing Pipeline:**
1. **Semantic Topic Search**: Uses sentence transformers to find related topics (threshold 0.5)
2. **Graph Traversal**: Expands from topic nodes through meaningful relationships
3. **Node Filtering**: Follows specific edge types (`HAS_TASK`, `RESPONSIBLE_TO`, `COLLABORATED_BY`, etc.)
4. **Visualization**: Generates interactive HTML graphs with hover details
5. **Natural Language**: LangGraph provides conversational answers grounded in graph data

**Graph Schema:**
```
Topic → [HAS_TASK] → Task → [RESPONSIBLE_TO] → Person → [HAS_ROLE] → Role → [BELONGS_TO] → Department → [IS_IN] → Organization
                    ↓
                [DUE_ON] → Date
                [BASED_ON] → Summary  
                [LINKED_TO] → Email Index
```

**Node Hierarchy Examples:**
- **Topic**: "Capstone project - Praxis"
- **Task**: "Sign NDA and share completed forms with the client team"
- **Person**: "Rachel Wang (Student Team Lead)"
- **Role**: "Student Team Lead (Praxis Capstone Project)"
- **Department**: "Praxis Capstone Project (University of California, San Diego)"
- **Organization**: "University of California, San Diego"

### **Data Flow Architecture**
```
Gmail Takeout → .mbox Upload → Smart Filtering → Email Parsing → 
LLM Extraction → HITL Validation → Graph Construction → 
Master Graph → GraphRAG Query → Interactive Visualization + NL Answers
```

---

## ✅ **Recently Implemented Improvements**

### **Graph Structure Enhancements**
- **Hierarchical Node Formatting**: Added `"Department (Organization)"` format to eliminate naming conflicts
- **Person Node Clarity**: Standardized to `"Name (Role)"` format for both owners and collaborators
- **Topic Variation Handling**: Enhanced semantic similarity + substring matching (threshold 0.5)
- **Node Expansion**: Optimized to 25 nodes for comprehensive results without over-expansion

### **User Experience Improvements**
- **Query Guidance**: Added comprehensive tips in AI Chatbot interface
- **Example Queries**: Show users how to include topic names for best results
- **Visual Consistency**: Reduced node and font sizes for better graph readability
- **Error Prevention**: Clear instructions prevent common query mistakes

### **System Reliability**
- **Large File Support**: Refactored to accept any size Inbox.mbox files - system processes first 200MB automatically
- **No ZIP Required**: Direct .mbox upload eliminates decompression errors and 403 issues  
- **Streamlit Configuration**: Set maxUploadSize=1GB to prevent browser file size rejections
- **Pandas Warning**: Fixed `SettingWithCopyWarning` in embedding pipeline by using `.copy()` to ensure proper DataFrame handling
- **HITL Validation**: Fully implemented human-in-the-loop JSON validation with pause/resume functionality  
- **Session State Management**: Robust handling of processing states in Streamlit UI

---

## 🔧 Large File Upload Troubleshooting

### Browser File Size Warnings
- **Issue**: Browser shows "File too large" or "200MB limit" error
- **Solution**: **Ignore the warning!** Our system is configured for large files
- **Why it happens**: Browser default warnings, but our backend handles files up to 1GB
- **What we process**: Only the first 200MB of your file for optimal performance

### Configuration Details
The app automatically handles large files through:
- **Backend config**: `maxUploadSize = 1GB` in `.streamlit/config.toml`
- **Smart processing**: Reads only first 200MB regardless of total file size  
- **Memory efficient**: Uses streaming to avoid loading entire file into memory
- **No ZIP required**: Direct .mbox upload eliminates decompression errors

### If Upload Still Fails
1. **Check file format**: Must be `.mbox` file (not ZIP)
2. **Try smaller chunks**: Extract date-filtered exports from Gmail
3. **Browser issues**: Try different browser (Chrome/Firefox work best)
4. **Network timeout**: Ensure stable internet connection during upload