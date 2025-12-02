FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY scheduler.py .
COPY report_generator.py .

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
