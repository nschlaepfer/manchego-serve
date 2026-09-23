# syntax=docker/dockerfile:1
#
# Manchego v2.1 on the System One wire contract, offline at run time.
#
# CUDA (default; linux/amd64; torch 2.10.0 built for CUDA 12.8; NVIDIA driver R570+ recommended, required on RTX 50-series):
#   docker build -t manchego-serve:2.1-cuda .
#   docker run --rm --gpus all -p 127.0.0.1:8000:8000 manchego-serve:2.1-cuda
#
# CPU (amd64 or arm64; float32 by default, needs ~20 GB RAM; add `--dtype bfloat16` to the run command for ~10 GB):
#   docker build -t manchego-serve:2.1-cpu \
#     --build-arg BASE_IMAGE=python:3.12.11-slim-bookworm \
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#
# Weights are downloaded ONCE, here, pinned by full commit sha and checked against the published SHA-256 of every
# weight file. Build with --build-arg DOWNLOAD_WEIGHTS=0 to leave them out and mount a folder at /models/manchego.

ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG DOWNLOAD_WEIGHTS=1
ARG MODEL_REPO=oraculumai/Manchego
ARG MODEL_REVISION=77403228b7dfdf823af99a5f562bdcf80b708d4c

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# torch: already in the CUDA base image; installed from the PyTorch wheel index otherwise. Either way it must be TORCH_VERSION.
RUN python -c "import torch" 2>/dev/null || pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
RUN python -c "import sys, torch; v = torch.__version__.split('+')[0]; sys.exit(0 if v == '${TORCH_VERSION}' else 'torch ' + torch.__version__ + ' is not ${TORCH_VERSION}')"

WORKDIR /opt/manchego-serve
COPY docker/requirements.txt docker/requirements.txt
RUN pip install -r docker/requirements.txt
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY manchego_serve manchego_serve
RUN pip install --no-deps . && python -c "import manchego_serve, manchego_serve.server"

RUN useradd --create-home --uid 10001 manchego && mkdir -p /models && chown manchego /models
USER manchego

RUN if [ "${DOWNLOAD_WEIGHTS}" = "1" ]; then \
      HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 manchego-serve-download --repo "${MODEL_REPO}" --revision "${MODEL_REVISION}" --out /models/manchego \
      && rm -rf /models/manchego/.cache; \
    else mkdir -p /models/manchego; fi

# Run time: no network. The server also forces these itself before importing any model code.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false \
    MANCHEGO_BACKEND=torch MANCHEGO_MODEL=/models/manchego MANCHEGO_REVISION=${MODEL_REVISION}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)" || exit 1
ENTRYPOINT ["manchego-serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
