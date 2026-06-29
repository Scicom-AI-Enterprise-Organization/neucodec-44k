#!/usr/bin/env bash
# Pod-side: resume the d20 finetune on the BIG (1500GB) pod, scaled with scale44k.
# One fire-and-forget trigger (SSH-drop safe):
#   1. pull the full last.ckpt (optimizer_states + lr_schedulers intact) from HF
#      into 44k_d20/last.ckpt  -> train.py auto-resumes (true continue from step 505k)
#   2. run_44k_finetune.sh: prepare ALL data (8 base corpora + 336 scale44k zips)
#      then launch the depth-20 finetune, which resumes from that last.ckpt.
set -uo pipefail
cd /neucodec-44k
set -a; source .env 2>/dev/null; set +a
# Fast HF downloads on the pod: keep Xet ENABLED (high-perf) — unlike local infer
# where xet hung, the pod's network handles it and it's much faster. HF_TOKEN
# (from .env) authenticates every download. US region keeps RTT to HF low.
export HF_HOME=/hf_cache HF_XET_HIGH_PERFORMANCE=1

mkdir -p /neucodec-44k/44k_d20
if [ ! -f /neucodec-44k/44k_d20/last.ckpt ]; then
  echo "===== [redeploy] pulling full last.ckpt (optimizer) from HF ====="
  .venv/bin/python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="Scicom-intl/neucodec-44k-d20", filename="last.ckpt",
                    repo_type="model", token=os.environ["HF_TOKEN"], local_dir="/tmp/ckptpull")
shutil.move(p, "/neucodec-44k/44k_d20/last.ckpt")
print("[redeploy] pulled -> /neucodec-44k/44k_d20/last.ckpt")
PY
else
  echo "[redeploy] 44k_d20/last.ckpt already present — skip pull"
fi

echo "===== [redeploy] data prep (8 base + scale44k) + resume d20 ====="
DEPTH=20 CKPT=null MAX_BATCHES=0 EPOCHS=10 \
  MOS_EVERY=10000 BATCH=48 ACCUM=1 NUM_WORKERS=16 \
  SOURCES="malay,sg,commonvoice,cartoons,movie,expresso_read,expresso_conv,ears,scale44k" \
  WANDB_NAME=44k_d20_scale44k LOG_DIR=/neucodec-44k/44k_d20 \
  bash runpod/run_44k_finetune.sh
