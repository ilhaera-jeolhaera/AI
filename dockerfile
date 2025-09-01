FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY cloudtype_requirements.txt .
RUN pip install --no-cache-dir -r cloudtype_requirements.txt

# Copy application files
COPY backend.py .
COPY load_data.py .
COPY attached_assets/ ./attached_assets/

# Copy ChromaDB data with proper permissions
COPY chroma_db_uiseong_20250831_new/ ./chroma_db_uiseong_20250831_new/
RUN chmod -R 755 ./chroma_db_uiseong_20250831_new/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False
ENV CHROMA_SERVER_NOFILE=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "backend.py"]