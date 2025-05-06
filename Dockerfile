## Dockerfile for Flask Recommendation Service
FROM python:3.9-slim

# Set up application environment
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
EXPOSE 8083 9100

ARG APP_VERSION=4.2.0
ENV APP_VERSION=${APP_VERSION}

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
COPY config.ini . 

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the application
CMD ["python", "app.py"]
