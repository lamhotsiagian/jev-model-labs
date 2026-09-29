# Decision service image. Build from the repo root:
#   docker build -f labs/ch07_service/Dockerfile -t jev-decision-service .
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JEV_BACKEND=sim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY jevkit ./jevkit
COPY labs ./labs
RUN useradd -m svc && mkdir -p var && chown svc var
USER svc
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/healthz')"
CMD ["uvicorn", "labs.ch07_service.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
