FROM python:3.13.5-slim

WORKDIR /app

COPY requirements-dev.txt .

RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src /app/src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]