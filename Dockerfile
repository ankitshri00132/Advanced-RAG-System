FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt .

RUN pip install --no-cache-dir -r requirements-dev.txt

COPY model_preload.py .

RUN python -m model_preload

COPY src /app/src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]