"""
Train TokenInterpolator for 25 TPS or 12 TPS.

The NeuCodec model is fully frozen. Only TokenInterpolator weights are trained.

Loss (combined):
  1. MSE on predicted 50 TPS embeddings vs true 50 TPS embeddings
     (weighted: interpolated positions * 1.0, known positions * 0.1)
  2. Multi-resolution mel reconstruction loss on decoded audio
     (gradients flow through frozen decoder ops back to interpolator)

Key insight: `requires_grad_(False)` on NeuCodec params only stops the
*parameters* from accumulating gradients. The forward activations still
carry gradients, so mel loss propagates through the frozen decoder to
the interpolator.

Usage:
    python train_interpolator.py \
        --ckpt path/to/neucodec.pt \
        --factor 2 \          # 2 = 25 TPS, 4 = 12 TPS
        --data /path/to/wavs \
        --output interp_25tps.pt
"""

import argparse
import glob
import random
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

from neucodec.model import NeuCodec
from neucodec.token_interpolator import TokenInterpolator
from criterions import MultiResolutionMelSpectrogramLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, sample_rate: int = 44_100, segment_secs: float = 3.0):
        self.files = glob.glob(f"{data_dir}/**/*.wav", recursive=True)
        assert len(self.files) > 0, f"No wav files found in {data_dir}"
        self.sr = sample_rate
        self.segment_len = int(segment_secs * sample_rate)
        self.segment_len = (self.segment_len // 882) * 882  # align to hop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)
        wav = wav.mean(0, keepdim=True)  # mono [1, T]

        if wav.shape[-1] >= self.segment_len:
            start = random.randint(0, wav.shape[-1] - self.segment_len)
            wav = wav[:, start:start + self.segment_len]
        else:
            wav = nn.functional.pad(wav, (0, self.segment_len - wav.shape[-1]))

        return wav.unsqueeze(0)  # [1, 1, T]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_50tps_embeddings(neucodec, audio):
    """
    Encode audio -> quantize -> lookup embeddings. Fully frozen, no grad.

    Returns:
        emb_50: [B, T, 1024]
    """
    codes = neucodec.encode_code(audio)                                           # [B, 1, T]
    emb = neucodec.generator.quantizer.get_output_from_indices(codes.transpose(1, 2))  # [B, T, 2048]
    emb = neucodec.fc_post_a(emb)                                                 # [B, T, 1024]
    return emb


def decode_embeddings(neucodec, emb):
    """
    Decode 50 TPS embeddings to audio via frozen backbone + ISTFT.
    Gradients DO flow through here (through ops, not params).

    Args:
        emb: [B, T, 1024]
    Returns:
        audio: [B, 1, T_audio]
    """
    audio, _ = neucodec.generator(emb, vq=False)
    return audio


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen NeuCodec
    print(f"Loading NeuCodec from {args.ckpt}")
    neucodec = NeuCodec._from_pretrained(
        model_id=None,
        local_ckpt_path=args.ckpt,
        map_location=device,
    )
    neucodec = neucodec.to(device).eval()
    for p in neucodec.parameters():
        p.requires_grad_(False)

    # Interpolator — only trainable module
    interp = TokenInterpolator(dim=1024, factor=args.factor, depth=args.depth, heads=8).to(device)
    print(f"TokenInterpolator params: {sum(p.numel() for p in interp.parameters()):,}")

    optimizer = torch.optim.AdamW(interp.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Mel loss (same sample rate and windows as main training)
    mel_loss_fn = MultiResolutionMelSpectrogramLoss(
        sample_rate=44_100,
        window_lengths=[128, 512, 2048],
        n_mels=[20, 80, 320],
    ).to(device)

    mse = nn.MSELoss()

    dataset = AudioDataset(args.data, segment_secs=args.segment_secs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    step = 0
    best_loss = float("inf")

    print(f"Training TokenInterpolator factor={args.factor} "
          f"({50 // args.factor} TPS -> 50 TPS) | lambda_mel={args.lambda_mel}")

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            audio_gt = batch.squeeze(0).to(device)  # [B, 1, T_audio]  44.1kHz

            # ----------------------------------------------------------------
            # 1. Get frozen ground-truth 50 TPS embeddings (no grad)
            # ----------------------------------------------------------------
            emb_50 = get_50tps_embeddings(neucodec, audio_gt)  # [B, T, 1024]

            # ----------------------------------------------------------------
            # 2. Subsample to low rate
            # ----------------------------------------------------------------
            emb_low = emb_50[:, ::args.factor, :].detach()  # [B, T//factor, 1024]

            # ----------------------------------------------------------------
            # 3. Interpolate back to 50 TPS (with grad)
            # ----------------------------------------------------------------
            emb_pred = interp(emb_low)  # [B, T_pred, 1024]

            T_min = min(emb_pred.shape[1], emb_50.shape[1])
            emb_pred_ = emb_pred[:, :T_min]
            emb_50_   = emb_50[:, :T_min].detach()

            # ----------------------------------------------------------------
            # 4. Embedding MSE loss
            #    - full weight on interpolated (missing) positions
            #    - small weight on known positions (identity regularization)
            # ----------------------------------------------------------------
            mask_missing = torch.ones(T_min, dtype=torch.bool, device=device)
            mask_missing[::args.factor] = False  # True = missing, False = known

            loss_interp = mse(emb_pred_[:, mask_missing], emb_50_[:, mask_missing])
            loss_known  = mse(emb_pred_[:, ~mask_missing], emb_50_[:, ~mask_missing])
            loss_emb = loss_interp + 0.1 * loss_known

            # ----------------------------------------------------------------
            # 5. Mel reconstruction loss
            #    Gradients flow: interp -> emb_pred -> frozen decoder -> audio -> mel
            # ----------------------------------------------------------------
            audio_pred = decode_embeddings(neucodec, emb_pred_)  # [B, 1, T_audio]

            # align lengths for mel comparison
            T_audio = min(audio_pred.shape[-1], audio_gt.shape[-1])
            loss_mel = mel_loss_fn(
                audio_pred[:, 0, :T_audio],
                audio_gt[:, 0, :T_audio],
            )

            # ----------------------------------------------------------------
            # 6. Combined loss
            # ----------------------------------------------------------------
            loss = loss_emb + args.lambda_mel * loss_mel

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(interp.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                print(
                    f"step {step:6d} | "
                    f"total={loss.item():.4f} "
                    f"emb={loss_emb.item():.4f} "
                    f"(interp={loss_interp.item():.4f} known={loss_known.item():.4f}) "
                    f"mel={loss_mel.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if step % 1000 == 0 and loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(interp.state_dict(), args.output)
                print(f"  -> saved best ({best_loss:.4f})")

    torch.save(interp.state_dict(), args.output)
    print(f"Done. Saved to {args.output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="NeuCodec .pt checkpoint")
    parser.add_argument("--factor", type=int, default=2, choices=[2, 4],
                        help="2 = 25 TPS, 4 = 12 TPS")
    parser.add_argument("--data", required=True, help="Directory with .wav files")
    parser.add_argument("--output", default="interp.pt")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=4,
                        help="Transformer depth in TokenInterpolator")
    parser.add_argument("--segment_secs", type=float, default=3.0)
    parser.add_argument("--lambda_mel", type=float, default=1.0,
                        help="Weight for mel reconstruction loss")
    args = parser.parse_args()
    train(args)
