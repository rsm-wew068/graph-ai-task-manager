# CI/CD & Deployment Guide

## 🚀 Continuous Integration & Deployment

This project includes comprehensive CI/CD pipelines using GitHub Actions.

## 📋 Workflows

### 1. **CI Pipeline** (`.github/workflows/ci.yml`)
Runs on every push and pull request to `main` and `develop` branches.

**Features:**
- ✅ Tests on Python 3.12 and 3.13
- 🔍 Code linting with flake8
- 🎨 Format checking with black
- 📝 Type checking with mypy
- 🐳 Docker image build and test
- 📦 Dependency caching for faster builds

### 2. **Deploy Pipeline** (`.github/workflows/deploy.yml`)
Triggers on:
- Pushes to `main` branch
- Version tags (`v*`)

**Features:**
- 🐳 Builds and pushes Docker images to Docker Hub
- 🤗 Auto-deploys to Hugging Face Spaces
- 🏷️ Semantic versioning support

### 3. **Security Scan** (`.github/workflows/security.yml`)
Runs on:
- Push to `main`/`develop`
- Pull requests
- Weekly schedule (Mondays at 9am UTC)

**Features:**
- 🔒 Dependency vulnerability scanning (Safety)
- 🛡️ Code security analysis (Bandit)
- 📊 Dependency review on PRs

## 🔐 Required Secrets

Add these secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

### Docker Hub Deployment
```
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password_or_token
```

### Hugging Face Spaces (Optional)
```
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_huggingface_username
```

## 🐳 Docker Deployment

### Local Development
```bash
# Start all services (app, PostgreSQL, Neo4j)
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Production Deployment
```bash
# Pull latest image
docker pull your_username/graph-ai-task-manager:latest

# Set environment variables in .env file
# Then run:
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Docker Build
```bash
# Build image
docker build -t graph-ai-task-manager .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e DATABASE_URL=your_db_url \
  -e NEO4J_URI=your_neo4j_uri \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=your_password \
  graph-ai-task-manager
```

## 🤗 Hugging Face Spaces Deployment

### Automatic (via GitHub Actions)
Pushes to `main` branch automatically deploy to Hugging Face Spaces.

### Manual Setup
1. Create a new Space on Hugging Face
2. Choose "Docker" as the SDK
3. Link to your GitHub repository or push code directly:
   ```bash
   git remote add hf https://huggingface.co/spaces/USERNAME/graph-ai-task-manager
   git push hf main
   ```

4. Add secrets in Space settings:
   - `OPENAI_API_KEY`
   - `DATABASE_URL`
   - `NEO4J_URI`
   - `NEO4J_USERNAME`
   - `NEO4J_PASSWORD`

## 🔄 Release Process

### Creating a Release
```bash
# Tag a version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

This triggers:
- Docker image build with version tags
- Automatic deployment to production

### Version Tags
- `v1.0.0` → Docker tags: `1.0.0`, `1.0`, `latest`
- `v1.2.3` → Docker tags: `1.2.3`, `1.2`, `latest`

## 🌐 Cloud Deployment Options

### AWS ECS/Fargate
```bash
# Use docker-compose.prod.yml
# Configure with AWS secrets manager for environment variables
```

### Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/graph-ai-task-manager

# Deploy
gcloud run deploy graph-ai-task-manager \
  --image gcr.io/PROJECT_ID/graph-ai-task-manager \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=xxx,DATABASE_URL=xxx
```

### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name graph-ai-task-manager \
  --image your_username/graph-ai-task-manager:latest \
  --dns-name-label graph-ai-task-manager \
  --ports 8501 \
  --environment-variables \
    OPENAI_API_KEY=xxx \
    DATABASE_URL=xxx
```

## 🔧 Environment Variables

### Required
- `OPENAI_API_KEY` - OpenAI API key for LLM processing
- `DATABASE_URL` - PostgreSQL connection string
- `NEO4J_URI` - Neo4j connection URI
- `NEO4J_USERNAME` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password

### Optional
- `LANGSMITH_API_KEY` - LangSmith tracing (debugging)
- `LANGCHAIN_TRACING_V2=true` - Enable LangSmith
- `PORT` - Custom port (default: 8501)

## 📊 Monitoring

### Health Checks
- App health: `http://localhost:8501/_stcore/health`
- Database: Automatic healthchecks in docker-compose

### Logs
```bash
# Docker logs
docker logs -f container_name

# Docker Compose logs
docker-compose logs -f

# Streamlit logs
tail -f logs/streamlit.log
```

## 🐛 Troubleshooting

### CI/CD Issues

**Build fails on import errors:**
- Check Python version compatibility
- Verify requirements.txt is up to date

**Docker build fails:**
- Check Dockerfile syntax
- Ensure all files are copied correctly
- Review .dockerignore

**Deployment fails:**
- Verify all secrets are set correctly
- Check environment variable names
- Review workflow logs in GitHub Actions

### Local Development

**Docker Compose issues:**
```bash
# Rebuild containers
docker-compose build --no-cache

# Reset volumes
docker-compose down -v
docker-compose up -d
```

## 📝 Best Practices

1. **Always test locally** before pushing to main
2. **Use feature branches** and pull requests
3. **Run linting** before committing: `black . && flake8 .`
4. **Update requirements.txt** when adding dependencies
5. **Tag releases** for production deployments
6. **Monitor security scans** weekly
7. **Keep secrets secure** - never commit to repository

## 🔗 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud)
