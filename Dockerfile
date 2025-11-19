FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install numpy BEFORE TensorFlow
RUN pip install --no-cache-dir numpy==1.26.4

# Install all other packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
