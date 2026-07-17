FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy_data/ deploy_data/
COPY app.py index.html ./

ENV PORT=8000
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
