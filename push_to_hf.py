#!/usr/bin/env python3
"""Push ONLY the finetuned NeuCodec-44k weights to a Hugging Face model repo.

Extracts the NeuCodec state dict from a PyTorch-Lightning checkpoint (strips the
`model.` prefix, drops the discriminators + optimizer state) and saves it as
`pytorch_model.bin` — the filename NeuCodec._from_pretrained downloads — so the
model loads with:

    NeuCodec._from_pretrained(model_id="Scicom-intl/neucodec-44k", decoder_depth=12)

Uploads the single weights file only — no source code.

  python push_to_hf.py --ckpt 44k/last.ckpt --repo Scicom-intl/neucodec-44k [--public]
"""
import argparse
import os

import torch
from huggingface_hub import HfApi


def convert(ckpt_path: str, out_path: str) -> int:
    print(f"[convert] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    print(f"[convert] {len(model_sd)} NeuCodec tensors (from {len(sd)} ckpt keys)")
    torch.save(model_sd, out_path)
    print(f"[convert] wrote {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)")
    return len(model_sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--code-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--public", action="store_true", help="default is a PRIVATE repo")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")

    out_bin = os.path.join(a.code_dir, "pytorch_model.bin")
    convert(a.ckpt, out_bin)

    api = HfApi(token=token)
    print(f"[hf] create_repo {a.repo} (private={not a.public})")
    api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)

    print("[hf] uploading weights (pytorch_model.bin) …")
    api.upload_file(path_or_fileobj=out_bin, path_in_repo="pytorch_model.bin",
                    repo_id=a.repo, repo_type="model")
    print(f"[hf] done (weights only) → https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
