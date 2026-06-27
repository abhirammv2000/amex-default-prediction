# Streamlit credit-risk analyst dashboard. Build context = repo root.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY serving/demo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the app imports app.* (serving/) and loads serving/artifacts/*
COPY serving/app ./serving/app
COPY serving/artifacts ./serving/artifacts
COPY serving/demo ./serving/demo

ENV PORT=8080
EXPOSE 8080
# Cloud Run injects $PORT; Streamlit must bind 0.0.0.0 and run headless.
CMD streamlit run serving/demo/streamlit_app.py \
    --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true \
    --server.enableCORS=false --server.enableXsrfProtection=false
