FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# minimale Pakete für psycopg / SQLAlchemy
RUN apt-get update && apt-get install -y \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Port für die API
EXPOSE 8000


CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
