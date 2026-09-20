FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/

ARG FLASH_ATTN_CUDA_ARCHS=90

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        ninja-build \
        sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/breeze-infer

COPY pyproject.toml uv.lock ./

RUN uv pip install --system setuptools wheel packaging ninja \
    && uv export --locked --format requirements.txt --no-emit-project > /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && torch_cuda_archs="$(printf '%s' "${FLASH_ATTN_CUDA_ARCHS}" | sed -E 's/([0-9])([0-9])/\1.\2/g; s/,/;/g')" \
    && MAX_JOBS=8 FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} TORCH_CUDA_ARCH_LIST=${torch_cuda_archs} uv pip install --system \
        --no-build-isolation \
        --no-deps \
        "flash-attn==2.8.3"

COPY . .

RUN python docker/smoke_check.py \
    && python -m pytest -q \
        tests/test_logits_process.py \
        tests/test_templates.py \
        tests/test_warmup_profile.py

ENTRYPOINT ["/opt/breeze-infer/docker/entrypoint.sh"]
CMD ["python", "-m", "breeze_infer.api", "--help"]
