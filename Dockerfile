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
COPY app.py ./
COPY utils/ ./utils/
COPY pages/ ./pages/
COPY .streamlit/ ./.streamlit/

# Make app executable
RUN chmod +x app.py

# Expose port for Streamlit
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Start the Streamlit app directly
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]