# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set the directory inside the container
WORKDIR /app

# 3. Install system dependencies for Sklearn and Numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]