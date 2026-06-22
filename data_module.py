import os
# Cap CPU thread pools before numpy/torch import (see train.py). Critical so the
# many dataloader workers don't each spawn one-pool-per-core and oversubscribe.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import random
import librosa
import soundfile as sf
import soxr
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils
from transformers import AutoFeatureExtractor
from tqdm import tqdm

# Files longer than this are windowed with ffmpeg (seek + decode just the slice);
# shorter files (the vast majority — CommonVoice clips) are read whole in-process
# with soundfile, which is ~40x faster than spawning ffmpeg per item.
LONG_FILE_SEC = 30.0


def _worker_init_fn(worker_id):
    """Pin each dataloader worker to a single CPU thread. The collate_fn runs the
    w2v-bert feature extractor (torch CPU ops), which otherwise defaults to
    num_cores threads PER worker — on a 224-core box that means workers x 224
    threads, catastrophic oversubscription that effectively hangs training.
    One thread/worker lets N workers actually run in parallel."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def ffmpeg_window(path, start, dur, sr):
    """Decode just [start, start+dur] of `path` to mono float32 at `sr` using
    ffmpeg input-seeking (`-ss` before `-i`), so long podcast files are never
    fully decoded. Returns a numpy float32 array (possibly shorter than dur*sr
    if the window runs past EOF) or None on failure."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}",
        "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1",
    ]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=30, check=True).stdout
        if not out:
            return None
        return np.frombuffer(out, dtype=np.float32).copy()
    except Exception:
        return None
class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = FSDataset(phase, self.cfg)
        # ds = FSDataset_add_STFT(phase, self.cfg)
        num_workers = self.cfg.dataset.get('num_workers', 5)
        prefetch_factor = self.cfg.dataset.get('prefetch_factor', 5)
        # pin_memory + persistent_workers + prefetch_factor overlap CPU data prep
        # (decode + feature extraction) and the async H2D copy with GPU compute;
        # worker_init_fn pins workers to 1 thread so they don't oversubscribe.
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor if num_workers > 0 else None,
                        collate_fn=ds.collate_fn,
                        worker_init_fn=_worker_init_fn,
                        pin_memory=True,
                        persistent_workers=num_workers > 0,
                        drop_last=(phase == 'train'))

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        pass

class FSDataset(Dataset):
    """Dataset batching wav, mel 
    and other acoustic features

    Args:
        phase: train, val, test
        cfg: hydra config
    """
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()
        
        self.sr = cfg.preprocess.audio.sr
        
        self.filelist = self.get_filelist(self.phase_cfg.filelist)
        self.min_audio_length = cfg.dataset.min_audio_length

    def __len__(self):
        return len(self.filelist)

    def get_filelist(self, fpath):
        """Each line is `path` or `path<TAB>duration_seconds` (written by
        prepare_data.py). The duration lets us pick a random window without
        probing the file in the hot loop."""
        items = []
        with open(fpath, 'r') as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                parts = l.split('\t')
                dur = 0.0
                if len(parts) > 1:
                    try:
                        dur = float(parts[1])
                    except ValueError:
                        dur = 0.0
                items.append((parts[0], dur))
        return items

    def __getitem__(self, idx):
        try:
            wavpath_full, duration = self.filelist[idx]
            need44 = int(self.min_audio_length / 16000 * 44100)
            window_sec = self.min_audio_length / 16000.0

            wav_44k = None
            if duration <= 0.0 or duration <= LONG_FILE_SEC:
                # Short file (the common case): read the whole thing in-process
                # with soundfile (fast), crop a random window at the source rate,
                # then resample with soxr (C, SIMD — sub-ms).
                wav, sr = sf.read(wavpath_full, dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                win_src = int(window_sec * sr)
                if len(wav) > win_src:
                    i = random.randint(0, len(wav) - win_src)
                    wav = wav[i:i + win_src]
                wav_44k = wav if sr == 44100 else soxr.resample(wav, sr, 44100)
                wav_16k = wav if sr == 16000 else soxr.resample(wav, sr, 16000)
            else:
                # Long podcast: seek + decode only the window with ffmpeg.
                start = random.uniform(0.0, duration - window_sec)
                wav_44k = ffmpeg_window(wavpath_full, start, window_sec + 0.1, 44100)
                if wav_44k is None or len(wav_44k) == 0:
                    raise RuntimeError("ffmpeg window decode returned empty")
                wav_16k = soxr.resample(wav_44k, 44100, 16000)

            wav_44k = np.asarray(wav_44k, dtype=np.float32)
            wav_16k = np.asarray(wav_16k, dtype=np.float32)

            # exact lengths (pad short / crop long); both come from the same
            # window so the 16k encoder input and 44.1k target stay aligned.
            if len(wav_44k) < need44:
                wav_44k = np.pad(wav_44k, (0, need44 - len(wav_44k)))
            wav_44k = wav_44k[:need44]
            if len(wav_16k) < self.min_audio_length:
                wav_16k = np.pad(wav_16k, (0, self.min_audio_length - len(wav_16k)))
            wav_16k = wav_16k[:self.min_audio_length]

            return {
                'wav': torch.from_numpy(wav_16k.copy()).float(),
                'wav_24k': torch.from_numpy(wav_44k.copy()).float(),
            }
        except Exception as e:
            print(f"[FSDataset] skip {self.filelist[idx]}: {e}")
            return None
    
    def collate_fn(self, bs):
        if not hasattr(self, '_feature_extractor'):
            self._feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        bs = [b for b in bs if b is not None]

        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)
        wavs_24k = [b['wav_24k'] for b in bs]
        wavs_24k = torch.stack(wavs_24k)

        # Extract features in main process (collate_fn runs in main process)
        # Process per-sample to match original shape: each returns (1, C, T), stack to (B, 1, C, T)
        feat_list = []
        for w in wavs:
            wav_pad = F.pad(w, (160, 160))
            feat = self._feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt").data['input_features']
            feat_list.append(feat)
        feats = torch.stack(feat_list)

        out = {
            'wav': wavs,
            'wav_24k': wavs_24k,
            'feats': feats,
        }
        return out

@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg):
 
    data_module = DataModule(cfg)

 
    train_loader = data_module.train_dataloader()

 
    valid_filelist = []

 
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing batches", unit="batch")):
 
        wavs = batch['wav']
 

if __name__ == "__main__":
    main()

