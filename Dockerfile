FROM python:3.12-alpine

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir uvicorn && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8080", "--threads", "8", "deadlock_analytics_api.main:app"]
