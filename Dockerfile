# syntax=docker/dockerfile:1
#
# Manchego on the System One wire contract, offline at run time. MANCHEGO_VERSION selects the model: v3 or v2.1.
# The default is v2.1 until this package pins v3's Hub revision (manchego_serve/weights.py, V3_REVISION: a placeholder
# until the v3 upload); then it becomes v3 (tests/test_weights.py fails until this default follows the pin).
#
# CUDA (default; linux/amd64; torch 2.10.0 built for CUDA 12.8; NVIDIA driver R570+ recommended, required on RTX 50-series):
#   docker build -t manchego-serve:2.1-cuda --build-arg MANCHEGO_VERSION=v2.1 .
#   docker build -t manchego-serve:3-cuda --build-arg MANCHEGO_VERSION=v3 .        # once V3_REVISION is pinned
#   docker run --rm --gpus all -p 127.0.0.1:8000:8000 manchego-serve:2.1-cuda
#
# CPU (amd64 or arm64; float32 by default, needs ~20 GB RAM; append `--dtype bfloat16` to the run command for ~10 GB):
#   docker build -t manchego-serve:2.1-cpu \
#     --build-arg BASE_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 \
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#   docker run --rm -p 127.0.0.1:8000:8000 manchego-serve:2.1-cpu --dtype bfloat16
#
# Arguments after the image name are appended to `manchego-serve`; the bind address and port come from MANCHEGO_HOST and
# MANCHEGO_PORT (0.0.0.0 and 8000 here), so extra arguments never drop them.
#
# Weights are downloaded ONCE, here, at the commit this package pins for MANCHEGO_VERSION (or at MODEL_REVISION, a full
# commit sha, when given) and checked against the pinned SHA-256 of every weight file and of the chat template, tokenizer
# and config files; the build fails on a mismatch. Build with --build-arg DOWNLOAD_WEIGHTS=0 to leave them out and mount
# a folder at /models/manchego (MANCHEGO_VERSION then only declares which version it holds).

ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG DOWNLOAD_WEIGHTS=1
ARG MANCHEGO_VERSION=v2.1
ARG MODEL_REPO=oraculumai/Manchego
ARG MODEL_REVISION=

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# the CUDA base image (Ubuntu 24.04) marks its system Python as externally managed (PEP 668); without this, pip refuses to install
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# torch: already in the CUDA base image; installed from the PyTorch wheel index otherwise. Either way it must be TORCH_VERSION.
RUN python -c "import torch" 2>/dev/null || pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
RUN python -c "import sys, torch; v = torch.__version__.split('+')[0]; sys.exit(0 if v == '${TORCH_VERSION}' else 'torch ' + torch.__version__ + ' is not ${TORCH_VERSION}')"

WORKDIR /opt/manchego-serve
COPY docker/requirements.txt docker/requirements.txt
RUN pip install -r docker/requirements.txt
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY manchego_serve manchego_serve
RUN pip install --no-deps --no-build-isolation . && python -c "import manchego_serve, manchego_serve.server"

RUN useradd --create-home --uid 10001 manchego && mkdir -p /models && chown manchego /models
USER manchego

# HF_HOME points at a scratch folder removed in the same layer, so no download or Xet cache stays in the image.
RUN if [ "${DOWNLOAD_WEIGHTS}" = "1" ]; then \
      HF_HOME=/tmp/hf-build HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 \
        manchego-serve-download --repo "${MODEL_REPO}" --revision "${MODEL_REVISION:-${MANCHEGO_VERSION}}" --out /models/manchego \
      && rm -rf /models/manchego/.cache /tmp/hf-build; \
    else mkdir -p /models/manchego; fi

# Run time: no network. The server also forces these itself before importing any model code. MANCHEGO_REVISION declares
# the build in /models/manchego: a version name stands for the commit this package pins (weights.py).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false \
    MANCHEGO_BACKEND=torch MANCHEGO_MODEL=/models/manchego MANCHEGO_REVISION=${MODEL_REVISION:-${MANCHEGO_VERSION}} \
    MANCHEGO_HOST=0.0.0.0 MANCHEGO_PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('MANCHEGO_PORT', '8000'), timeout=4)" || exit 1
ENTRYPOINT ["manchego-serve"]
