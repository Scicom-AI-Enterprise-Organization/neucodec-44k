#!/usr/bin/env bash
# Pod-side: add Expresso + EARS + cartoons to /data, rebuild the filelist, then
# resume d20 training (uncapped) on the expanded dataset. One fire-and-forget
# trigger so flaky proxied-SSH can't interrupt it mid-launch.
set -uo pipefail
cd /neucodec-44k
set -a; source .env 2>/dev/null; set +a
export HF_HOME=/hf_cache HF_HUB_DISABLE_XET=1 HF_XET_HIGH_PERFORMANCE=1

echo "===== [expand] cleaning stale procs ====="
pkill -9 -f prepare_data.py 2>/dev/null || true
pkill -9 -f run_44k_finetune 2>/dev/null || true
pkill -9 -f "/neucodec-44k/.venv/bin/python train.py" 2>/dev/null || true
rm -rf /data/expresso_read /data/expresso_conv   # re-export cleanly
sleep 3

echo "===== [expand] add sources + rebuild filelist ====="
.venv/bin/python prepare_data.py --data-root /data \
    --sources expresso_read,expresso_conv,ears,cartoons
echo "[expand] train files now: $(wc -l < /data/train.txt)"

echo "===== [expand] resume d20 training on expanded data ====="
# run_44k_finetune.sh skips data prep (train.txt exists) and resumes 44k_d20/last.ckpt
DEPTH=20 CKPT=/neucodec-44k/extended_20.pt MAX_BATCHES=0 EPOCHS=10 \
  MOS_EVERY=10000 BATCH=48 ACCUM=1 NUM_WORKERS=16 \
  WANDB_NAME=44k_d20_expanded LOG_DIR=/neucodec-44k/44k_d20 \
  bash runpod/run_44k_finetune.sh
