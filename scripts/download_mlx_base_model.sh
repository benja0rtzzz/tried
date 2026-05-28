#!/usr/bin/env bash
# Authenticate with HuggingFace and download the MLX 4-bit base model
# (mlx-community/Qwen2.5-Coder-14B-4bit) into ~/models/qwen-coder-14b-4bit.
#
# Run from the repo root on the MacBook:
#   bash scripts/download_mlx_base_model.sh
#
# HF_TOKEN resolution order:
#   1. HF_TOKEN already exported in the shell
#   2. HF_TOKEN= line in packages/orchestrator/.env
#   3. Existing cached login from a previous `hf auth login`
#   4. Interactive `hf auth login` prompt (last resort)

set -euo pipefail

MODEL_ID="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
LOCAL_DIR="${HOME}/models/qwen-coder-14b-instruct-4bit"
ENV_FILE="packages/orchestrator/.env"

# --- locate HF_TOKEN -------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" && -f "${ENV_FILE}" ]]; then
  token_line="$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | tail -1 || true)"
  if [[ -n "${token_line}" ]]; then
    HF_TOKEN="${token_line#HF_TOKEN=}"
    HF_TOKEN="${HF_TOKEN%\"}"; HF_TOKEN="${HF_TOKEN#\"}"
    HF_TOKEN="${HF_TOKEN%\'}"; HF_TOKEN="${HF_TOKEN#\'}"
    export HF_TOKEN
    echo "[auth] loaded HF_TOKEN from ${ENV_FILE}"
  fi
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[auth] using HF_TOKEN from environment"
  uv run --package tried-orchestrator hf auth login --token "${HF_TOKEN}" --add-to-git-credential
elif uv run --package tried-orchestrator hf auth whoami >/dev/null 2>&1; then
  user="$(uv run --package tried-orchestrator hf auth whoami 2>/dev/null | head -1)"
  echo "[auth] reusing existing cached login (${user})"
else
  echo "[auth] no HF_TOKEN found and no cached login — launching interactive login"
  echo "       (paste a token from https://huggingface.co/settings/tokens)"
  uv run --package tried-orchestrator hf auth login
fi

# --- download --------------------------------------------------------------

mkdir -p "${LOCAL_DIR}"
echo "[download] ${MODEL_ID} -> ${LOCAL_DIR}"
uv run --package tried-orchestrator hf download \
  "${MODEL_ID}" \
  --local-dir "${LOCAL_DIR}"

echo
echo "[done] model is at ${LOCAL_DIR}"
echo "Next: export TRIED_MLX_CHECKPOINT=${LOCAL_DIR}"
