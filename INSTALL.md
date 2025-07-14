# Installation Guide

## Quick Start

### Option 1: Using uv (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd graph-ai-task-manager

# Install dependencies using uv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your Neo4j and OpenAI credentials

# Start the application
python app.py
```

### Option 2: Using pip (Alternative)
```bash
# Install in development mode
pip install -e .

# Start the application
python app.py
```

## Environment Variables

Create a `.env` file with the following variables:

```env
# Neo4j Database
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI API
OPENAI_API_KEY=your_openai_api_key
```

## Troubleshooting

### Import Error: No module named 'langgraph'

This error occurs when dependencies aren't properly installed. Try:

1. **Upgrade pip**: `pip install --upgrade pip`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Check Python version**: Ensure you're using Python 3.12+
4. **Virtual environment**: Use a clean virtual environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Neo4j Connection Issues

1. **Start Neo4j**: Ensure Neo4j is running on the specified URI
2. **Check credentials**: Verify username/password in `.env`
3. **Test connection**: 
   ```bash
   python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); print('Connected!')"
   ```

### OpenAI API Issues

1. **Check API key**: Ensure `OPENAI_API_KEY` is set in `.env`
2. **Verify quota**: Check your OpenAI account has available credits
3. **Test API**: 
   ```bash
   python -c "import openai; openai.api_key='your_key'; print('API key valid!')"
   ```

## Docker Deployment

```bash
# Build the image
docker build -t task-manager .

# Run with environment variables
docker run -p 8501:8501 -p 8000:8000 \
  -e NEO4J_URI=bolt://host.docker.internal:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=password \
  -e OPENAI_API_KEY=your_key \
  task-manager
```

## Services

After starting the application:

- **Streamlit UI**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **Neo4j Browser**: http://localhost:7474 (if running locally)

## Dependencies

Core dependencies include:
- `streamlit>=1.28.0` - Web UI
- `langgraph>=0.2.31` - Workflow orchestration
- `langchain>=0.3.26` - LLM framework
- `neo4j>=5.28.1` - Graph database
- `sentence-transformers>=5.0.0` - Embeddings
- `fastapi>=0.116.1` - API backend 