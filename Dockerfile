# ----------------------------------------------------
# 1. BASE IMAGE
# ----------------------------------------------------
FROM python:3.10-slim

# ----------------------------------------------------
# 2. Install system dependencies (compatible with TF)
# ----------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# 3. Set working directory
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# 4. Copy backend files
# ----------------------------------------------------
COPY . .

# ----------------------------------------------------
# 5. Install Python requirements
# ----------------------------------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# 6. Expose the Render port
# ----------------------------------------------------
ENV PORT=10000
EXPOSE 10000

# ----------------------------------------------------
# 7. Start the server using Gunicorn
# ----------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
