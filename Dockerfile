# Use Python 3.10 – compatible with TensorFlow 2.12
FROM python:3.10-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install pip dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Cloud Run port
ENV PORT=8080

# Run the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
