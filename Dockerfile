FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory with proper permissions for HF Spaces
RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /tmp && chmod 777 /tmp

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all necessary files and directories
COPY app.py ./
COPY utils/ ./utils/
COPY pages/ ./pages/
COPY .streamlit/ ./.streamlit/

# Expose port 8501 (consistent with streamlit config)
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH="/app:${PYTHONPATH}"

# For Hugging Face Spaces compatibility - use dynamic port if available
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]