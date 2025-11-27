# This is the base image we will be using: a lightweight Python image
FROM python:3.12-slim 

# Set working directory
WORKDIR /app

# Install system dependencies: we need gcc for pytorch
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY submission/ ./submission/

# Copy environment files - you MUST include these in your submission
COPY pyproject.toml .

# Copy example evaluation files
COPY model_calls.py .
COPY utils.py .

# Install UV package manager
RUN pip install uv

# Install dependencies
RUN uv sync

# Run the evaluation script - DO NOT CHANGE THIS
CMD ["uv", "run", "python", "model_calls.py"]

