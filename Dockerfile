FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ✅ Only copy requirements.txt first (to cache pip install layer)
COPY requirements.txt .

# ✅ Install dependencies first — only runs again if requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Then copy the full source code
COPY . .

# Run Streamlit app
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0