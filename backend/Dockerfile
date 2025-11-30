# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY ./app /app/app
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y build-essential gcc --no-install-recommends \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get remove -y build-essential gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

