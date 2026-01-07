FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/market_recorder/requirements.txt ./services/market_recorder/

# Install Python dependencies
RUN pip install --no-cache-dir -r services/market_recorder/requirements.txt

# Copy application code
COPY services/ ./services/
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "services.market_recorder.main:app", "--host", "0.0.0.0", "--port", "8000"]
