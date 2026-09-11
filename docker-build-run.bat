@echo off
echo ====================================================================
echo CivicFlow AI - Single Container Build & Deployment Helper
echo ====================================================================

echo 1. Building Docker image 'civicflow-ai:latest'...
docker build -t civicflow-ai:latest .

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker build failed. Please ensure Docker Desktop is running.
    exit /b %ERRORLEVEL%
)

echo 2. Removing existing container 'civicflow-ai-app' if running...
docker rm -f civicflow-ai-app 2>nul

echo 3. Starting container 'civicflow-ai-app' on port 8000...
docker run -d --name civicflow-ai-app -p 8000:8000 civicflow-ai:latest

if %ERRORLEVEL% EQU 0 (
    echo ====================================================================
    echo [SUCCESS] CivicFlow AI container is running!
    echo REST Endpoints:
    echo   - Health check: http://localhost:8000/health
    echo   - DAG Pipeline: POST http://localhost:8000/api/v1/ai/pipeline
    echo   - Swagger Docs: http://localhost:8000/docs
    echo ====================================================================
) else (
    echo [ERROR] Failed to launch Docker container.
)
