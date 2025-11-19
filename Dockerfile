# Use Python 3.10 Slim
FROM python:3.10-slim

# -------------------------------
# Install required system packages
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Copy requirements and install
# -------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# TensorFlow 2.10.0 (works on CPU x86)
RUN pip install tensorflow==2.10.0

# Install other Python packages
RUN pip install -r requirements.txt

# -------------------------------
# Copy backend source code
# -------------------------------
COPY . .

# -------------------------------
# Expose Render port
# -------------------------------
ENV PORT=10000

# Start server
CMD gunicorn --bind 0.0.0.0:$PORT app:app
