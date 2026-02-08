FROM python:3.11-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -m nltk.downloader punkt && \
    python -m nltk.downloader punkt_tab

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
