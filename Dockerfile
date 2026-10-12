
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80 \
    MODEL_PATH=/app/models/model.bin

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:80/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]