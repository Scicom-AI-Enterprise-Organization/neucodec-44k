import os

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import random
import librosa
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils
from transformers import AutoFeatureExtractor
from tqdm import tqdm
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
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        collate_fn=ds.collate_fn,
                        pin_memory=True,
                        persistent_workers=True)

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
        with open(fpath, 'r') as f:
            flist = [l.strip().split('\t')[0] for l in f if l.strip()]
        return flist

    def __getitem__(self, idx):
        try:
            wavpath_full = self.filelist[idx]
            min_audio_length_24k = int(self.min_audio_length / 16000 * 44100)

            # Load at native sr using librosa (returns numpy)
            original_wav, sr = librosa.load(wavpath_full, sr=None, mono=True)

            # Resample to 16kHz for encoder input
            if sr != 16000:
                wav_16k = librosa.resample(original_wav, orig_sr=sr, target_sr=16000)
            else:
                wav_16k = original_wav

            length = len(wav_16k)
            if length < self.min_audio_length:
                wav_16k = np.pad(wav_16k, (0, self.min_audio_length - length))
                length = len(wav_16k)
            i = random.randint(0, length - self.min_audio_length)
            wav_16k = wav_16k[i:i + self.min_audio_length]

            wav_16k = torch.from_numpy(wav_16k).float()

            # Resample to 44.1kHz for decoder target
            if sr != 44100:
                wav_48k = librosa.resample(original_wav, orig_sr=sr, target_sr=44100)
            else:
                wav_48k = original_wav

            length = len(wav_48k)
            if length < min_audio_length_24k:
                wav_48k = np.pad(wav_48k, (0, min_audio_length_24k - length))
                length = len(wav_48k)
            i = random.randint(0, length - min_audio_length_24k)
            wav_48k = wav_48k[i:i + min_audio_length_24k]

            wav_48k = torch.from_numpy(wav_48k).float()

            out = {
                'wav': wav_16k,
                'wav_24k': wav_48k,
            }

            return out
        except Exception as e:
            print(e)
    
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

