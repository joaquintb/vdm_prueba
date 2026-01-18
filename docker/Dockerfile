# CUDA-capable PyTorch base image (falls back to CPU if no GPU is available)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# (Optional) If using BuildKit, you can speed up pip installs with:
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements-docker.txt

# Install Python deps (copy first to leverage Docker layer cache)
COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements-docker.txt

# Copy project code
COPY . /app

# Default command: run the full pipeline (can be overridden at docker run time)
CMD ["python", "-m", "src.pipeline", "--force"]