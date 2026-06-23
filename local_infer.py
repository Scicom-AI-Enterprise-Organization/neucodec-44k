#!/usr/bin/env python3
"""Encode->decode every audio file in ./audio with the finetuned NeuCodec-44k
checkpoint and write the 44.1 kHz reconstructions back into ./audio.

  HF_TOKEN=... python3 local_infer.py
"""
import glob
import os

# Xet downloads can hang on some networks — force plain HTTPS for HF downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import librosa
import soundfile as sf
import torch

from neucodec import NeuCodec

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
REPO = os.environ.get("NEUCODEC_REPO", "Scicom-intl/neucodec-44k")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[infer] loading {REPO} (decoder_depth=12) on {device} …")
model = NeuCodec._from_pretrained(
    model_id=REPO, decoder_depth=12, token=os.environ.get("HF_TOKEN")
).eval().to(device)
sr_out = model.sample_rate
print(f"[infer] model sample_rate = {sr_out}")

files = sorted(
    f for f in glob.glob(os.path.join(AUDIO_DIR, "*"))
    if f.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg"))
    and "_recon44k" not in f
)
print(f"[infer] {len(files)} input file(s)")

for f in files:
    # load mono @16kHz for the (frozen) encoder; encode -> codes -> decode -> 44.1kHz
    wav16, _ = librosa.load(f, sr=16000, mono=True)
    x = torch.from_numpy(wav16).float().view(1, 1, -1).to(device)
    with torch.no_grad():
        codes = model.encode_code(x)
        wav = model.decode_code(codes).squeeze().detach().cpu().float().numpy()
    out = os.path.splitext(f)[0] + "_recon44k.wav"
    sf.write(out, wav, sr_out)
    info = sf.info(out)
    print(f"[infer] {os.path.basename(f)} -> {os.path.basename(out)} "
          f"| codes={tuple(codes.shape)} | {len(wav)/sr_out:.2f}s @ {info.samplerate} Hz")

print("[infer] done.")
