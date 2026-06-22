#!/usr/bin/env bash
# Drive the full NeuCodec-44k run ON THE POD: download+extract data, then launch
# the decoder finetune with per-epoch MOS. Resumable. Everything under / .
#
# Tunables via env (defaults in brackets):
#   EPOCHS [10]  BATCH [8]  NUM_WORKERS [16]  ACCUM [2]  SOURCES [malay,sg,commonvoice]
set -euo pipefail

REPO=/neucodec-44k
VENV=$REPO/.venv
cd "$REPO"

# Secrets / caches — keep HF + wandb entirely off /workspace.
set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a
export HF_HOME=/hf_cache
export HF_XET_HIGH_PERFORMANCE=1   # datasets are Xet-backed
export TOKENIZERS_PARALLELISM=false
# Base model + w2v-bert are cached after the first run; go offline so the 32
# dataloader workers don't each fire HF HEAD requests for the feature extractor.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Critical on this 224-core box: without this, each dataloader worker's torch
# CPU ops spawn ~224 threads -> workers x 224 threads oversubscribe and hang.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /data /hf_cache "$REPO/44k"

EPOCHS=${EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:--1}          # >0 caps by step
MAX_TIME=${MAX_TIME:-}              # "DD:HH:MM:SS" wall-clock cap (best for a fixed budget)
MOS_EVERY=${MOS_EVERY:-10000}       # MOS every N true batches (0 = at epoch end)
BATCH=${BATCH:-8}
NUM_WORKERS=${NUM_WORKERS:-16}   # with 1 thread/worker the pipeline does ~9k batches/s; 16 is plenty
ACCUM=${ACCUM:-1}
SOURCES=${SOURCES:-malay,sg,commonvoice}

# Reduce CUDA fragmentation so large batches don't OOM on transient peaks.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Limiter priority: wall-clock (MAX_TIME) > step-cap (MAX_STEPS) > epochs.
if [ -n "$MAX_TIME" ]; then
    LIMIT_OVERRIDES="train.trainer.max_steps=-1 +train.trainer.max_epochs=10000 +train.trainer.max_time=$MAX_TIME"
elif [ "$MAX_STEPS" -gt 0 ] 2>/dev/null; then
    LIMIT_OVERRIDES="train.trainer.max_steps=$MAX_STEPS +train.trainer.max_epochs=10000"
else
    LIMIT_OVERRIDES="train.trainer.max_steps=-1 +train.trainer.max_epochs=$EPOCHS"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    "$VENV/bin/python" -m wandb login "$WANDB_API_KEY" || true
fi

# ---- 1. data: download + extract + filelists (skip if already built) --------
if [ ! -f /data/train.txt ]; then
    echo "===== [run] preparing data (sources=$SOURCES) ====="
    "$VENV/bin/python" prepare_data.py --data-root /data --sources "$SOURCES"
else
    echo "===== [run] /data/train.txt exists — skipping data prep ====="
fi
echo "[run] train files: $(wc -l < /data/train.txt)  | mos files: $(wc -l < /data/mos.txt)"

# ---- 2. finetune: decoder-only -> 44.1kHz, FSQ codebook frozen, per-epoch MOS
echo "===== [run] launching finetune: ${EPOCHS} epochs, batch ${BATCH}x${ACCUM}, 1x H100 ====="
exec "$VENV/bin/python" train.py \
    log_dir="$REPO/44k" \
    wandb_name=44k \
    wandb_project=neucodec_44k \
    every_n_train_steps=5000 \
    save_top_k=3 \
    ckpt=null \
    train.trainer.devices=1 \
    ~train.trainer.min_steps \
    $LIMIT_OVERRIDES \
    train.trainer.limit_val_batches=0 \
    ~train.trainer.val_check_interval \
    +train.accumulate_grad_batches="$ACCUM" \
    dataset.train.filelist=/data/train.txt \
    dataset.val.filelist=/data/test.txt \
    dataset.test.filelist=/data/test.txt \
    dataset.train.batch_size="$BATCH" \
    dataset.num_workers="$NUM_WORKERS" \
    dataset.prefetch_factor=4 \
    mos.enable=true \
    mos.filelist=/data/mos.txt \
    mos.every_n_steps="$MOS_EVERY"
