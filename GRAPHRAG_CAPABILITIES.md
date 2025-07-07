# GraphRAG Question Capabilities

## Overview
The task-focused GraphRAG system uses **semantic task name matching + focused graph expansion** to provide highly accurate results. The system now prioritizes finding tasks by semantic similarity to their names, then expands to show related context - resulting in much higher accuracy and relevance.

## Two-Stage Query Process

### 🎯 **Stage 1: Semantic Task Name Matching (High Accuracy)**
- **How it works**: Uses AI embeddings to semantically match your query against actual task names
- **Threshold**: 0.3 (finds semantically related tasks)  
- **Result**: High confidence scores (0.4-0.9) when task names match your intent
- **Example**: "coaching appointment" → finds "Reschedule coaching appointment" (0.835 confidence)

### 🔍 **Stage 2: General Semantic Search (Fallback)**  
- **When used**: Only when no task names match semantically
- **Scope**: Searches all node types (people, topics, dates, tasks)
- **Result**: Broader results with lower confidence scores
- **Use case**: General exploration when you're not sure about specific task names

## Question Types Supported

### 👤 Person-Focused Queries
Ask about specific people and their work:
- "What tasks does Rachel have?"
- "Show me assignments for John Smith"
- "Who is working on AI projects?"
- "What is Sarah responsible for?"
- "Find all tasks assigned to team members"

### 📊 Topic & Project Queries
Explore projects and topics:
- "Tell me about AI-Assisted Customer Analytics"
- "What projects involve machine learning?"
- "Show me marketing initiatives"
- "Information about customer segmentation"
- "What research projects are ongoing?"

### 📅 Date & Timeline Queries
Find information by time periods:
- "What tasks are due in February 2025?"
- "Show me deadlines this month"
- "What events are coming up?"
- "Tasks due before March"
- "Show me this week's schedule"

### ⚡ Status & Urgency Queries
Focus on priority and status:
- "What tasks are overdue?"
- "Show me urgent assignments"
- "What needs immediate attention?"
- "High priority items"
- "Tasks requiring quick action"

### 🏢 Organization & Department Queries
Explore by organizational structure:
- "Show me MSBA department assignments"
- "What work is the analytics team doing?"
- "Display university projects"
- "Marketing department tasks"
- "Research group activities"

### 🎯 Specific Event/Task Queries
Get details about particular items:
- "Details about the TRACE event"
- "Information on customer segmentation project"
- "Show me presentation tasks"
- "Meeting preparation items"
- "Conference planning details"

### 🔍 General Exploration
Broad discovery queries:
- "What people are involved in this project?"
- "What are the main topics?"
- "Show me all active work"
- "What's happening this week?"
- "Give me an overview of current tasks"

## How It Works

### 1. Semantic Search
- Converts the user's question into embeddings
- Finds the most semantically relevant nodes in the knowledge graph
- No need for exact keyword matches

### 2. Graph Expansion
- Starting from the most relevant nodes found
- Explores connected nodes to gather comprehensive context
- Includes related people, dates, tasks, and topics

### 3. Flexible Response
- Returns all relevant information with confidence scores
- Works with the LLM to generate natural language answers
- Adapts to different question phrasings and intentions

## Key Advantages

✅ **No Rigid Patterns**: Unlike the previous complex system, there are no hardcoded query patterns or entity-type detection that could fail

✅ **Natural Language**: Users can ask questions in their own words without learning specific syntax - both casual and formal language work

✅ **Focused Results**: Max 10 nodes prevents information overload and solves the "hub node problem" where shared dates would pull in unrelated tasks

✅ **Fast Performance**: Optimized for real-time responses with focused graph expansion

✅ **Flexible Confidence**: The system provides confidence scores but doesn't reject questions based on artificial thresholds

✅ **Extensible**: As new data is added to the knowledge graph, the system automatically becomes capable of answering questions about it

✅ **Semantic Understanding**: Uses AI embeddings to understand meaning, not just keyword matching

## Best Practices for High Accuracy

### ✅ **Include Task-Related Keywords:**
- **Instead of**: "What does Rachel have?" 
- **Try**: "Rachel's coaching appointment" or "Rachel's job update tasks"

### ✅ **Use Task Name Concepts:**
- **Instead of**: "Show me work"
- **Try**: "evaluation tasks", "meeting preparation", "application work"

### ✅ **Specific Over General:**  
- **High accuracy**: "12Twenty job status" → finds exact task
- **Lower accuracy**: "career stuff" → falls back to general search

### ✅ **Natural Task Descriptions:**
- "coaching appointment" ✅
- "job status update" ✅  
- "peer review work" ✅
- "office hours meeting" ✅

## Natural Language Flexibility

### ✅ **Multiple Ways to Ask the Same Thing:**
- **Formal**: "What tasks does Rachel have?" 
- **Casual**: "Rachel's tasks" or "work for Rachel"
- **Variations**: "assignments Rachel" or "show me Rachel's work"

### ✅ **Different Phrasings Supported:**
- **Deadlines**: "upcoming deadlines", "due dates", "when things are due"
- **Projects**: "AI work", "machine learning stuff", "analytics projects" 
- **Status**: "what's urgent?", "what needs attention?", "high priority items"
- **People**: "who's doing what?", "team members working on...", "people involved"

## Testing
The system has been tested with dozens of different question types and consistently returns relevant results. The demo script `demo_graphrag_questions.py` can be run to see live examples of the system's capabilities.

## Integration
The GraphRAG works seamlessly with the LangGraph pipeline, providing context that the LLM uses to generate helpful, accurate responses to user questions about their tasks and projects.
