# 1. Use an official Python runtime
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy project files into the container
COPY . /app

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose port 8000 for FastAPI
EXPOSE 8000

# 6. Command to run FastAPI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
