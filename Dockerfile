FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
