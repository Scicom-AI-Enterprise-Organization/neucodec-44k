#!/usr/bin/env bash
# Pod-side: add malaysian-movie-youtube (48k) to /data, rebuild the filelist, then
# resume d20 training on it. One fire-and-forget trigger (SSH-drop safe).
set -uo pipefail
cd /neucodec-44k
set -a; source .env 2>/dev/null; set +a
export HF_HOME=/hf_cache HF_HUB_DISABLE_XET=1 HF_XET_HIGH_PERFORMANCE=1

echo "===== [movie] stop training ====="
pkill -9 -f prepare_data.py 2>/dev/null || true
pkill -9 -f run_44k_finetune 2>/dev/null || true
pkill -9 -f "/neucodec-44k/.venv/bin/python train.py" 2>/dev/null || true
sleep 3

echo "===== [movie] download + extract + rebuild filelist ====="
.venv/bin/python prepare_data.py --data-root /data --sources movie
echo "[movie] train files now: $(wc -l < /data/train.txt)"

echo "===== [movie] resume d20 on movie-expanded data ====="
DEPTH=20 CKPT=/neucodec-44k/extended_20.pt MAX_BATCHES=0 EPOCHS=10 \
  MOS_EVERY=10000 BATCH=48 ACCUM=1 NUM_WORKERS=16 \
  WANDB_NAME=44k_d20_movie LOG_DIR=/neucodec-44k/44k_d20 \
  bash runpod/run_44k_finetune.sh
