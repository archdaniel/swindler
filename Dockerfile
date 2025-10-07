# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- Dependency Installation ---
# Install essential build tools and the missing libgomp1 library

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
libgomp1 \
&& rm -rf /var/lib/apt/lists/*

# --->>> END ADDED SECTION <<<---
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Install system dependencies if needed (e.g., for certain ML libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Code & Data ---
# Copy the entire project context into the container
COPY ./src ./src
COPY ./models ./models
COPY ./data/generated ./data/generated

# --- NEW: Set PYTHONPATH ---
# Add the WORKDIR to the PYTHONPATH so Python can find packages inside ./src
ENV PYTHONPATH="${PYTHONPATH}:/app"

# --- Expose Port ---
# ... (previous lines) ...

# --- Expose Port ---
EXPOSE 7860
# --- Run Application ---
CMD ["uvicorn", "src.smart_collections_api.main:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]
