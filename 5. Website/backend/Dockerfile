
# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Set working directory inside Docker
WORKDIR /app

# 3. Copy backend requirements
COPY backend/requirements.txt ./requirements.txt

# 4. Install backend dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# 5. Copy FastAPI code and trained ML model
COPY backend/ ./backend/

# 6. Document FastAPI port
EXPOSE 8000

# 7. Start FastAPI
CMD ["uvicorn", "backend.api_code.main:app", "--host", "0.0.0.0", "--port", "8000"]
