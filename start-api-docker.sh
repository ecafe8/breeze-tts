#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_path="${MODEL_PATH:-$repo_dir/ckpts/breeze-tts-2}"
image_tag="${BREEZE_IMAGE:-breeze-pytorch-infer:latest}"
port="${PORT:-7860}"

if [[ ! -d "$model_path" ]]; then
  echo "Model directory not found: $model_path" >&2
  echo "Set MODEL_PATH or download the checkpoint to ckpts/breeze-tts-2." >&2
  exit 1
fi

if ! docker image inspect "$image_tag" >/dev/null 2>&1; then
  echo "Docker image not found: $image_tag" >&2
  echo "Build it first with: FLASH_ATTN_CUDA_ARCHS=89 bash docker/build.sh" >&2
  exit 1
fi

api_args=(
  --host 0.0.0.0
  --port "$port"
)

if [[ "${FAST_ALL:-0}" == "1" ]]; then
  api_args+=(--fast-all)
fi

echo "Breeze TTS API: http://0.0.0.0:$port"
echo "LAN health check: http://<host-ip>:$port/health"

docker run --rm --gpus all \
  --ipc=host \
  --publish "$port:$port" \
  --volume "$model_path:/models/breeze:ro" \
  "$image_tag" \
  python -m breeze_infer.api /models/breeze \
  "${api_args[@]}" \
  "$@"
