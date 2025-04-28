# Use official slim Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment PORT to 8501 as default
ENV PORT 8501

# Run the Streamlit app binding to PORT
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
