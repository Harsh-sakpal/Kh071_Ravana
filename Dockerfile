# ==============================================================================
# CivicFlow AI - Unified Single Production Container
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/civicflow-ai-person2 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ai /app/ai
COPY civicflow-ai-person2 /app/civicflow-ai-person2
COPY pipeline /app/pipeline
COPY tests /app/tests

# Expose HTTP port
EXPOSE 8000

# Native Docker Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set working directory to microservice root for Uvicorn execution
WORKDIR /app/civicflow-ai-person2

# Entrypoint command running production Uvicorn REST service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
