# Multi-stage build: install deps into a venv, then copy into a slim runtime.
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY serving/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY serving/app ./app
COPY serving/artifacts ./artifacts

# Drop privileges
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')" || exit 1
# Shell form so Cloud Run's injected $PORT is expanded (defaults to 8080).
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
