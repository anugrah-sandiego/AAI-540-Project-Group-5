FROM python:3.10-slim

WORKDIR /app

# Show logs live in Docker/CloudWatch; LOG_LEVEL and LOG_FORMAT (text or json)
# are read by src/utils/logging_config.py
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=text

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create MLflow tracking directory
RUN mkdir -p mlruns

# Expose port for MLflow UI (if needed)
EXPOSE 5000

# Default command
CMD ["python", "-m", "src.pipelines.training_pipeline"]
