#!/usr/bin/env python3
"""Score a directory of wavs with UTMOSv2 — predicted naturalness MOS.

Runs in the ISOLATED MOS venv (utmosv2 has heavy, version-pinned deps that we
keep out of the training venv). The per-epoch callback invokes this as a
subprocess and parses the `@@MOS {json}` line off stdout.

API mirrors the Scicom eval (gateway/.../tts/tts_eval.py:score_mos) verbatim:
    utmosv2.create_model(pretrained=True).predict(input_path=…, num_repetitions=5)

Usage:
    python mos_score.py --wav-dir /path/to/wavs [--num-repetitions 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--num-repetitions", type=int, default=5)
    a = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(a.wav_dir, "*.wav")))
    if not wavs:
        log(f"[mos] no wavs in {a.wav_dir}")
        print('@@MOS {"mos": null, "n": 0}', flush=True)
        return

    import utmosv2  # heavy import — only here

    log(f"[mos] loading UTMOSv2; scoring {len(wavs)} file(s) …")
    model = utmosv2.create_model(pretrained=True)

    scores = []
    for w in wavs:
        try:
            s = float(model.predict(input_path=w, num_repetitions=a.num_repetitions))
            scores.append(s)
        except Exception as e:  # noqa: BLE001
            log(f"[mos] skip {w}: {e}")

    mean = sum(scores) / len(scores) if scores else None
    print(f"@@MOS {json.dumps({'mos': mean, 'n': len(scores)})}", flush=True)
    log(f"[mos] mean MOS={mean} over {len(scores)} file(s)")


if __name__ == "__main__":
    main()
