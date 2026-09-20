#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_path="${MODEL_PATH:-$repo_dir/ckpts/breeze-tts-2}"
host="${HOST:-127.0.0.1}"
port="${PORT:-7860}"

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$TRANSFORMERS_CACHE"
  unset TRANSFORMERS_CACHE
fi

if [[ ! -d "$model_path" ]]; then
  echo "Model directory not found: $model_path" >&2
  echo "Set MODEL_PATH or download the checkpoint to ckpts/breeze-tts-2." >&2
  exit 1
fi

api_args=(
  "$model_path"
  --host "$host"
  --port "$port"
)

if [[ "${FAST_ALL:-0}" == "1" ]]; then
  api_args+=(--fast-all)
fi

if ! uv run python -c "import flash_attn" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Warning: flash-attn is not installed; inference will use the slower PyTorch attention path.
FlashAttention 2.8.3 must be built with the same CUDA version as PyTorch (12.8).
Use the CUDA 12.8 Docker image with:
  FLASH_ATTN_CUDA_ARCHS=89 bash docker/build.sh
For a local install, first install the CUDA 12.8 toolkit and ninja, then set
CUDA_HOME to that toolkit before running the FlashAttention build command.
EOF
fi

exec uv run python -m breeze_infer.api "${api_args[@]}" "$@"
