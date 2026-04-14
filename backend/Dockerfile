FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir . && rm -rf /root/.cache/pip

RUN mkdir -p /app/data/chroma_db /app/data/dev_cache /app/data/code_rag_vectors

EXPOSE 8923

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8923"]
