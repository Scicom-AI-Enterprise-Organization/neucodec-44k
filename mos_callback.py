"""Per-epoch MOS evaluation callback for the NeuCodec-44k decoder finetune.

At the end of every training epoch (rank 0 only) it:
  1. takes a fixed held-out set of audio files (cfg.mos.filelist),
  2. runs each through the *current* model: encode_code -> decode_code, i.e. the
     exact reconstruction path being finetuned (codebook/encoder frozen, only
     the 44.1kHz decoder changes),
  3. writes the reconstructed wavs (and ground-truth refs) to disk,
  4. scores them with UTMOSv2 via mos_score.py in the isolated MOS venv,
  5. logs `val/mos` (reconstruction) and `val/mos_gt` (ground-truth ceiling) to
     wandb, keyed by epoch.

UTMOSv2 is reference-free, so this measures predicted naturalness of what the
decoder produces — the metric the user asked to track each epoch. The scorer
call mirrors gateway/.../tts/tts_eval.py exactly.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning.callbacks import Callback

from data_module import ffmpeg_window


def _read_filelist(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            dur = float(parts[1]) if len(parts) > 1 else 0.0
            items.append((parts[0], dur))
    return items


class MOSEvalCallback(Callback):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.mos
        self.enable = bool(m.get("enable", True))
        self.filelist = m.get("filelist", None)
        # With this dataset an "epoch" is ~272k steps, so measure MOS on a step
        # cadence instead. every_n_steps>0 → step-based; ==0 → at epoch end.
        self.every_n_steps = int(m.get("every_n_steps", 0))
        self.max_samples = int(m.get("max_samples", 64))
        self.window_sec = float(m.get("window_sec", 10.0))
        self.num_repetitions = int(m.get("num_repetitions", 5))
        self.out_dir = m.get("out_dir", os.path.join(cfg.log_dir, "mos_eval"))
        self.mos_python = m.get("python", "python")  # MOS venv interpreter
        self.mos_script = m.get("script", "mos_score.py")
        self.score_gt = bool(m.get("score_gt", True))
        self.sr = 44100
        self._items = None
        self._last_step = -1

    # -- helpers ----------------------------------------------------------------
    def _load_items(self):
        if self._items is None:
            if not self.filelist or not os.path.exists(self.filelist):
                self._items = []
            else:
                self._items = _read_filelist(self.filelist)[: self.max_samples]
        return self._items

    def _score_dir(self, wav_dir):
        cmd = [self.mos_python, self.mos_script, "--wav-dir", wav_dir,
               "--num-repetitions", str(self.num_repetitions)]
        # The training run sets HF_HUB_OFFLINE=1, but UTMOSv2 needs to fetch its
        # SSL backbone (facebook/wav2vec2-base) on first use — let the scorer go
        # online (it caches into HF_HOME, so later calls are instant).
        env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
        try:
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=3600, env=env).stdout.decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            print(f"[mos] scorer subprocess failed: {e}", flush=True)
            return None
        mos = None
        for line in out.splitlines():
            if line.startswith("@@MOS "):
                try:
                    mos = json.loads(line[len("@@MOS "):]).get("mos")
                except json.JSONDecodeError:
                    pass
            else:
                print(line, flush=True)
        return mos

    @torch.no_grad()
    def _reconstruct(self, pl_module, gen_dir, ref_dir):
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(ref_dir, exist_ok=True)
        model = pl_module.model
        was_training = model.training
        model.eval()
        n = 0
        for i, (path, dur) in enumerate(self._load_items()):
            win = min(self.window_sec, dur) if dur > 0 else self.window_sec
            wav16 = ffmpeg_window(path, 0.0, win, 16000)
            if wav16 is None or len(wav16) < 1600:
                continue
            try:
                x = torch.from_numpy(wav16.copy()).float().view(1, 1, -1).to(pl_module.device)
                codes = model.encode_code(x)
                recon = model.decode_code(codes).squeeze().detach().cpu().float().numpy()
                sf.write(os.path.join(gen_dir, f"{i:04d}.wav"), recon, self.sr)
                if self.score_gt:
                    ref = ffmpeg_window(path, 0.0, win, self.sr)
                    if ref is not None and len(ref):
                        sf.write(os.path.join(ref_dir, f"{i:04d}.wav"), ref, self.sr)
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[mos] reconstruct skip {path}: {e}", flush=True)
        if was_training:
            model.train()
        return n

    # -- hooks ------------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0 or not self.enable or not trainer.is_global_zero:
            return
        # Fire on TRUE batch count (pl_module._train_batches), not trainer.global_step
        # (which double-counts the two optimizers). So MOS lands on exact 20k-batch marks.
        b = getattr(pl_module, "_train_batches", trainer.global_step)
        if b > 0 and b % self.every_n_steps == 0 and b != self._last_step:
            self._last_step = b
            self._run_eval(trainer, pl_module, tag=f"batch{b:08d}")

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_steps > 0 or not self.enable or not trainer.is_global_zero:
            return
        self._run_eval(trainer, pl_module, tag=f"epoch{trainer.current_epoch:03d}")

    def _run_eval(self, trainer, pl_module, tag):
        items = self._load_items()
        if not items:
            print("[mos] no mos filelist / empty — skipping MOS eval", flush=True)
            return
        gen_dir = os.path.join(self.out_dir, tag, "gen")
        ref_dir = os.path.join(self.out_dir, tag, "ref")
        print(f"[mos] {tag}: reconstructing {len(items)} sample(s) …", flush=True)
        n = self._reconstruct(pl_module, gen_dir, ref_dir)
        if n == 0:
            print("[mos] produced no audio to score — skipping", flush=True)
            return

        mos = self._score_dir(gen_dir)
        logs = {"epoch": trainer.current_epoch,
                "batch": getattr(pl_module, "_train_batches", trainer.global_step)}
        if mos is not None:
            logs["val/mos"] = mos
            print(f"[mos] {tag}: val/mos = {mos:.4f}", flush=True)
        if self.score_gt and os.listdir(ref_dir):
            mos_gt = self._score_dir(ref_dir)
            if mos_gt is not None:
                logs["val/mos_gt"] = mos_gt
                print(f"[mos] {tag}: val/mos_gt = {mos_gt:.4f} (ground-truth ceiling)", flush=True)

        # Log straight to the wandb run to avoid DDP sync (rank-0 only metric).
        try:
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.log(logs, step=trainer.global_step)
        except Exception as e:  # noqa: BLE001
            print(f"[mos] wandb log failed: {e}", flush=True)
