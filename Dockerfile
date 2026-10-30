FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt
COPY . /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/data/hf_cache
EXPOSE 8000
CMD ["uvicorn", "ipindia_corpus.search_api:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
