# Base Python Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]