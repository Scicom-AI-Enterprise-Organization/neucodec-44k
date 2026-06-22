#!/usr/bin/env python3
"""Standalone DataLoader smoke test — exercises the FULL data path (workers +
collate_fn incl. the w2v-bert feature extractor), with NO model/trainer, to
confirm the pipeline produces batches and how fast.

  HF_HUB_OFFLINE=1 .venv/bin/python test_dataloader.py [num_workers] [n_batches]
"""
import sys
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_module import FSDataset

num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 12

cfg = OmegaConf.create({
    "dataset": {"train": {"filelist": "/data/train.txt", "batch_size": 8, "shuffle": True},
                "min_audio_length": 96000},
    "preprocess": {"audio": {"sr": 16000}},
})
ds = FSDataset.__new__(FSDataset)
ds.phase = "train"; ds.cfg = cfg; ds.phase_cfg = cfg.dataset.train
ds.sr = 16000; ds.min_audio_length = 96000
ds.filelist = ds.get_filelist("/data/train.txt")
print(f"items: {len(ds.filelist)}  num_workers={num_workers}", flush=True)

dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=num_workers,
                prefetch_factor=4, collate_fn=ds.collate_fn,
                pin_memory=True, persistent_workers=True)

print("creating iterator (spawns workers)...", flush=True)
t0 = time.time()
it = iter(dl)
print(f"iterator ready in {time.time() - t0:.1f}s; pulling batches...", flush=True)

times = []
for i in range(n_batches):
    t = time.time()
    b = next(it)
    dt = time.time() - t
    times.append(dt)
    print(f"batch {i}: {dt:.2f}s  wav={tuple(b['wav'].shape)} "
          f"wav44={tuple(b['wav_24k'].shape)} feats={tuple(b['feats'].shape)}", flush=True)

steady = times[2:] or times
print(f"steady-state: {sum(steady)/len(steady)*1000:.0f} ms/batch "
      f"=> {len(steady)/sum(steady):.1f} batches/s", flush=True)
print("DATALOADER OK", flush=True)
