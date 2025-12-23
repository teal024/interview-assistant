FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime deps:
# - ffmpeg: required by faster-whisper to decode browser audio (WebM/Opus)
# - libsndfile1: required by python-soundfile (used by /tts)
# - libgomp1: required by ctranslate2 (used by faster-whisper)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

ENV DATABASE_URL=sqlite+aiosqlite:////data/data.db \
    HF_HOME=/data/hf \
    XDG_CACHE_HOME=/data/cache \
    NUMBA_CACHE_DIR=/data/numba

VOLUME ["/data"]

EXPOSE 8000

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
