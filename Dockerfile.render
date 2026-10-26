# Leaner image for Render web services. Prefer this over the default Dockerfile
# on Render (excludes PySpark / pytest / ruff / mypy).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GUARDIANEYE_HOST=0.0.0.0 \
    GUARDIANEYE_PORT=8742 \
    PORT=8742

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-render.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY deploy ./deploy

RUN pip install --upgrade pip \
    && pip install -r requirements-render.txt \
    && pip install --no-deps -e .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/models /app/reports \
    && chown -R appuser:appuser /app \
    && chmod +x /app/deploy/render/entrypoint.sh

USER appuser

RUN python scripts/generate_sample_data.py \
    && python scripts/prepare_image_data.py --config configs/data.yaml \
    && python scripts/prepare_audio_data.py --config configs/data.yaml \
    && python scripts/build_paired_events.py --config configs/data.yaml

EXPOSE 8742

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import os,urllib.request; p=os.environ.get('PORT','8742'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health')" || exit 1

CMD ["/app/deploy/render/entrypoint.sh"]
