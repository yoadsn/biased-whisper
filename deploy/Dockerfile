FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BW_DATA=/tmp/bwhisper-data
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . python-multipart
COPY deploy ./deploy
RUN test -s deploy/assets/model/model.safetensors && test -s deploy/assets/digits/adapter.safetensors && test -s deploy/assets/cardinals/adapter.safetensors && test -s deploy/assets/presets/manifest.json && test "$(find deploy/assets/presets -name 'sample-*.wav' | wc -l)" -eq 20
EXPOSE 8080
CMD ["uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
