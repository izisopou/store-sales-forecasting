# 1. Use official lightweight Python 3.12 image
FROM python:3.12-slim

# 2. Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Install OpenMP shared library required by XGBoost and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set working directory inside the container
WORKDIR /app

# 5. Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy pipeline source code and SQL queries
COPY sql/ ./sql/
COPY src/ ./src/
COPY main.py .

# 7. Create root marker file so pyprojroot.here() resolves /app as project root
RUN touch .project-root

# 8. Set main.py as the default container entry point
ENTRYPOINT ["python", "main.py"]