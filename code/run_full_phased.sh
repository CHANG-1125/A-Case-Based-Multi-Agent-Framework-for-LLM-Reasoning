#!/usr/bin/env bash
# 分阶段实验：先 zeroshot+rag，再 ours（同一 JSON，第二阶段需 --resume 合并）。
# 默认 500 题（省时间）；全量 test 设 FULL_TEST=1 或 NUM_SAMPLES=1319 且勿用 --full_test 时看下方说明。
set -euo pipefail
cd "$(dirname "$0")"

# Hugging Face Hub：避免实验性 httpx 在 SSL/代理失败后触发 “client has been closed”
export HF_HUB_DISABLE_EXPERIMENTAL_HTTPX="${HF_HUB_DISABLE_EXPERIMENTAL_HTTPX:-1}"
# 嵌入模型已在本机缓存时，禁止联网校验（缓解 UNEXPECTED_EOF / 不稳定网络）
export EMBEDDING_LOCAL_FILES_ONLY="${EMBEDDING_LOCAL_FILES_ONLY:-1}"

SEED="${SEED:-42}"
OUT="${OUT:-results/phased_500.json}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"

if [[ "${FULL_TEST:-0}" == "1" ]]; then
  SAMPLE_ARGS=(--full_test)
  echo "[samples] FULL_TEST=1 -> entire GSM8K test split"
else
  SAMPLE_ARGS=(--num_samples "${NUM_SAMPLES}")
  echo "[samples] NUM_SAMPLES=${NUM_SAMPLES} (set FULL_TEST=1 for full test)"
fi

# 若想从旧小样本扩到更大样本（如 300->500），可在手动命令中加 --resume_extend。
# 脚本默认阶段2只做同题数续跑，不做扩容。
# shellcheck source=/dev/null
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

echo "=== Phase 1: zeroshot + rag -> ${OUT}"
python run_experiments.py \
  "${SAMPLE_ARGS[@]}" \
  --methods zeroshot,rag \
  --seed "${SEED}" \
  --checkpoint_every 10 \
  --output "${OUT}"

echo "=== Phase 2: ours (resume merge) -> ${OUT}"
python run_experiments.py \
  "${SAMPLE_ARGS[@]}" \
  --methods ours \
  --seed "${SEED}" \
  --resume \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --output "${OUT}"

echo "Done. Results: ${OUT}"
