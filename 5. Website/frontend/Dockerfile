
# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Set the working directory inside Docker
WORKDIR /app

# 3. Copy frontend requirements
COPY frontend/requirements.txt ./requirements.txt

# 4. Install frontend dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# 5. Copy Streamlit and feature engineering code
COPY frontend/ ./frontend/

# 6. Document the Streamlit port
EXPOSE 8501

# 7. Start Streamlit
CMD ["streamlit", "run", "frontend/frontend.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
