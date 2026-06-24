#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for the NeuCodec-44k decoder finetune.
# Runs as root on the pod. Everything lives under / (the container disk),
# NEVER /workspace (the network volume). Idempotent enough to re-run.
set -euo pipefail

REPO=/neucodec-44k
VENV=$REPO/.venv          # training venv
MOSVENV=$REPO/.venv-mos   # isolated UTMOSv2 venv
export DEBIAN_FRONTEND=noninteractive

echo "===== [bootstrap] system packages ====="
apt-get update -y
apt-get install -y --no-install-recommends \
    ffmpeg unzip p7zip-full git rsync curl wget ca-certificates aria2 || true

# Static 7zz (matches the dataset READMEs; handles multi-volume zips reliably).
if ! command -v 7zz >/dev/null 2>&1; then
    echo "[bootstrap] installing 7zz static binary"
    cd /tmp
    wget -q https://www.7-zip.org/a/7z2301-linux-x64.tar.xz -O 7z.tar.xz && \
        tar -xf 7z.tar.xz 7zz && mv 7zz /usr/local/bin/ && chmod +x /usr/local/bin/7zz || \
        echo "[bootstrap] 7zz download failed; will fall back to p7zip's 7z"
fi

echo "===== [bootstrap] uv ====="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# Training venv. Install torch/torchaudio FIRST from the cu128 index (matches
# the base image's CUDA), THEN everything else from PyPI only. Doing torch in a
# separate step keeps the pytorch index from polluting the resolution of pure
# PyPI packages (it otherwise drags in an ancient numba/llvmlite with no py3.12
# wheels). torch is already satisfied so step 2 won't touch it.
# ---------------------------------------------------------------------------
echo "===== [bootstrap] training venv ($VENV) ====="
uv venv "$VENV" --python 3.12
# torchao must come from the cu128 index too (torchtune imports it at startup,
# and it must match torch's CUDA build).
uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchaudio==2.8.0 torchao
uv pip install --python "$VENV/bin/python" \
    numpy "numba>=0.61" "llvmlite>=0.44" \
    "pytorch-lightning>=2.2" hydra-core omegaconf \
    "transformers>=4.44.2" einops librosa soundfile soxr scipy wandb tqdm mutagen \
    "huggingface_hub>=0.34" hf_transfer hf_xet datasets \
    "vector-quantize-pytorch>=1.17.8" torchtune "local_attention>=1.11.1"

echo "[bootstrap] training venv torch/CUDA check:"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"

# ---------------------------------------------------------------------------
# MOS venv: UTMOSv2. Try the Scicom faster fork (needs GITHUB_TOKEN), then the
# public sarulab build (same `utmosv2` API + metric). MOS is best-effort — the
# callback tolerates this venv being absent, so training never blocks on it.
# ---------------------------------------------------------------------------
echo "===== [bootstrap] MOS venv ($MOSVENV) ====="
uv venv "$MOSVENV" --python 3.11
MOS_OK=0
if [ -n "${GITHUB_TOKEN:-}" ]; then
    uv pip install --python "$MOSVENV/bin/python" \
        "git+https://${GITHUB_TOKEN}@github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2" \
        soundfile librosa && MOS_OK=1 || echo "[bootstrap] faster-UTMOSv2 install failed"
fi
if [ "$MOS_OK" -eq 0 ]; then
    uv pip install --python "$MOSVENV/bin/python" \
        "git+https://github.com/sarulab-speech/UTMOSv2.git" soundfile librosa \
        && MOS_OK=1 || echo "[bootstrap] sarulab UTMOSv2 install failed (MOS will be skipped)"
fi
if [ "$MOS_OK" -eq 1 ]; then
    "$MOSVENV/bin/python" -c "import utmosv2; print('utmosv2 import OK')" || \
        echo "[bootstrap] WARNING: utmosv2 import failed; MOS eval will be skipped"
    # Pre-warm UTMOSv2's weights + its facebook/wav2vec2-base SSL backbone into the
    # SAME cache (/hf_cache) the training-time scorer uses, so the first MOS eval is
    # instant and works even though training runs HF_HUB_OFFLINE=1.
    echo "[bootstrap] pre-warming UTMOSv2 model cache ..."
    HF_HOME=/hf_cache CUDA_VISIBLE_DEVICES="" "$MOSVENV/bin/python" -c \
        "import utmosv2; utmosv2.create_model(pretrained=True); print('utmosv2 prewarm OK')" || \
        echo "[bootstrap] WARNING: utmosv2 prewarm failed (first MOS will download on demand)"
fi

echo "===== [bootstrap] done ====="
