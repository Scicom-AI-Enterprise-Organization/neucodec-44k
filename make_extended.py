#!/usr/bin/env python3
"""Build an extended NeuCodec init for the decoder-depth ablation.

Downloads the BASE neucodec weights (12 decoder layers) and replicates the last
decoder transformer layer to reach `--target-depth`, so the new layers start
warm (a copy of a trained layer) rather than random. Saves a raw NeuCodec state
dict loadable via `train.py ckpt=extended_N.pt model.codec_decoder.depth=N`.

  python make_extended.py --target-depth 16 --output extended_16.pt
"""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS — Xet can hang

import torch
from huggingface_hub import hf_hub_download

PREFIX = "generator.backbone.transformers."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="neuphonic/neucodec")
    ap.add_argument("--target-depth", type=int, required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    path = hf_hub_download(repo_id=a.model_id, filename="pytorch_model.bin",
                           token=os.environ.get("HF_TOKEN"))
    sd = torch.load(path, map_location="cpu")

    idxs = {int(k[len(PREFIX):].split(".")[0]) for k in sd if k.startswith(PREFIX)}
    cur = max(idxs) + 1
    last = cur - 1
    print(f"[extend] base decoder depth={cur}; target={a.target_depth}")
    if a.target_depth <= cur:
        print("[extend] target <= current; saving unchanged")
    else:
        last_keys = {k: v for k, v in sd.items() if k.startswith(f"{PREFIX}{last}.")}
        for new in range(cur, a.target_depth):
            for k, v in last_keys.items():
                sd[k.replace(f"{PREFIX}{last}.", f"{PREFIX}{new}.")] = v.clone()
            print(f"[extend]   layer {last} -> {new}")

    torch.save(sd, a.output)
    print(f"[extend] wrote {a.output} ({os.path.getsize(a.output)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
