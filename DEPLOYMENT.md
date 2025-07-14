# Deployment Guide

## GitHub Repository Setup

This repository is designed for GitHub deployment. Here are the deployment options:

## Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-ai-task-manager.git
cd graph-ai-task-manager

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Start the application
python start_app.py
```

## Docker Deployment

### Local Docker
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

### Docker Compose (Recommended)
Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  app:
    build: .
    ports:
      - "8501:8501"
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - neo4j

volumes:
  neo4j_data:
```

Run with:
```bash
docker-compose up -d
```

## Cloud Deployment

### Railway
1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push

### Render
1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python app.py`
5. Add environment variables

### Heroku
1. Create a `Procfile`:
```
web: python app.py
```
2. Deploy using Heroku CLI or GitHub integration
3. Set environment variables in Heroku dashboard

### Google Cloud Run
1. Build and push to Google Container Registry
2. Deploy to Cloud Run with environment variables
3. Set up automatic deployments from GitHub

## Environment Variables

Required environment variables for deployment:

```env
# Neo4j Database
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI API
OPENAI_API_KEY=your_openai_api_key
```

## CI/CD

The repository includes GitHub Actions for:
- Automated testing
- Dependency validation
- FastAPI endpoint testing
- Streamlit app testing

## Monitoring

- **Streamlit UI**: http://your-domain:8501
- **FastAPI Backend**: http://your-domain:8000
- **Neo4j Browser**: http://your-neo4j-host:7474

## Troubleshooting

### Port Issues
- Ensure ports 8501 and 8000 are exposed
- Check firewall settings
- Verify container networking

### Database Connection
- Test Neo4j connection separately
- Verify credentials and network access
- Check Neo4j logs for connection issues

### API Key Issues
- Verify OpenAI API key is valid
- Check API quota and billing
- Test API key separately 