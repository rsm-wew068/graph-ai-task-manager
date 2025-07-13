FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create cache directory with proper permissions
RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /tmp && chmod 777 /tmp

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy all necessary files and directories
COPY app.py start_services.py ./
COPY utils/ ./utils/
COPY pages/ ./pages/
COPY .streamlit/ ./.streamlit/

# Make scripts executable
RUN chmod +x app.py start_services.py

# Expose ports for both services
EXPOSE 8501 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Use the service orchestrator that runs both services
CMD ["python", "start_services.py"]