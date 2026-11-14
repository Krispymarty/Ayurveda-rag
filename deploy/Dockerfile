FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
WORKDIR /app
COPY deploy/runtime-lock.txt /tmp/runtime-lock.txt
# Same Torch version as the protected Linux baseline, CPU distribution only.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 3 --no-deps torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 3 -r /tmp/runtime-lock.txt && pip check
COPY . /app
RUN useradd --system --uid 10001 --create-home ayurip \
    && chmod -R a-w /app
USER 10001:10001
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/app/data/hf_cache \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AYURIP_DEPLOYMENT_MODE=public_research PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=70s --retries=3 \
    CMD python -c "import os,urllib.request,urllib.parse; r=urllib.request.Request('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',headers={'Host':urllib.parse.urlsplit(os.environ['AYURIP_PUBLIC_ORIGIN']).netloc}); urllib.request.urlopen(r,timeout=8)"
CMD ["python", "-B", "-m", "tools.cloud_serve"]
