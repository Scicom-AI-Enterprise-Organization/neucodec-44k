"""
TokenInterpolator: Upsample low-rate (25/12 TPS) token embeddings back to 50 TPS.

The original 50 TPS NeuCodec codebook is completely frozen and unchanged.
This module operates purely in embedding space (after quantizer lookup).

Encode at 25 TPS:
    audio -> NeuCodec.encode_code() -> 50 TPS codes [B, 1, T]
          -> take every 2nd token   -> 25 TPS codes [B, 1, T//2]

Decode from 25 TPS:
    25 TPS codes -> quantizer.get_output_from_indices -> 25 TPS embeddings [B, T//2, 1024]
                -> TokenInterpolator(factor=2) -> 50 TPS embeddings [B, T, 1024]
                -> NeuCodec decoder backbone + ISTFT -> audio

Training:
    Freeze entire NeuCodec. Only train TokenInterpolator.
    Loss: MSE on the predicted (odd-position) embeddings vs the true 50 TPS embeddings.
    Optional: reconstruction loss via frozen decoder.
"""

import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings
from .bs_roformer5 import TransformerBlock


class TokenInterpolator(nn.Module):
    """
    Upsamples from low-rate token embeddings to 50 TPS embeddings.

    Args:
        dim:    embedding dimension (1024, matching fc_post_a output)
        factor: upsample factor — 2 for 25->50 TPS, 4 for 12->50 TPS
        depth:  number of transformer layers
        heads:  attention heads
    """

    def __init__(self, dim: int = 1024, factor: int = 2, depth: int = 4, heads: int = 8):
        super().__init__()
        assert factor in (2, 4), "factor must be 2 (25 TPS) or 4 (12 TPS)"
        self.factor = factor
        self.dim = dim

        # Learned sub-position embeddings to distinguish slots within each group.
        # e.g. factor=2: slot 0 = known token, slot 1 = to be predicted.
        self.sub_pos_embed = nn.Embedding(factor, dim)

        rotary_embed = RotaryPositionalEmbeddings(dim=64)
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim=dim, n_heads=heads, rotary_embed=rotary_embed)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_low, dim]  — embeddings at 25 or 12 TPS

        Returns:
            out: [B, T_low * factor, dim]  — embeddings at 50 TPS
        """
        B, T, D = x.shape

        # Repeat each embedding `factor` times along time axis
        # [B, T, D] -> [B, T, factor, D] -> [B, T*factor, D]
        x = x.unsqueeze(2).expand(B, T, self.factor, D).reshape(B, T * self.factor, D)

        # Add sub-position embedding so the model knows which slot it's filling.
        # sub_idx: [0,1,0,1,...] for factor=2; [0,1,2,3,0,1,2,3,...] for factor=4
        sub_idx = torch.arange(self.factor, device=x.device).repeat(T)  # [T*factor]
        x = x + self.sub_pos_embed(sub_idx)  # broadcast over batch

        x = self.transformer(x)
        x = self.norm(x)
        return x  # [B, T*factor, D]


def encode_low_rate(neucodec, audio, factor: int = 2) -> torch.Tensor:
    """
    Encode audio to low-rate codes.

    Returns:
        codes: [B, 1, T//factor]  integer token indices
    """
    codes = neucodec.encode_code(audio)          # [B, 1, T] at 50 TPS
    codes = codes[:, :, ::factor]                # [B, 1, T//factor]
    return codes


def decode_low_rate(neucodec, interpolator: TokenInterpolator, codes: torch.Tensor) -> torch.Tensor:
    """
    Decode low-rate codes back to 48kHz audio via interpolation.

    Args:
        codes: [B, 1, T_low]  — 25 or 12 TPS codes

    Returns:
        audio: [B, 1, T_audio]  — 48kHz audio
    """
    # 1. Lookup embeddings for the known tokens [B, T_low, 2048]
    emb = neucodec.generator.quantizer.get_output_from_indices(codes.transpose(1, 2))
    # 2. Project to 1024-dim space [B, T_low, 1024]
    emb = neucodec.fc_post_a(emb)

    # 3. Interpolate to 50 TPS [B, T_high, 1024]
    emb = interpolator(emb)

    # 4. Decode with existing frozen backbone + ISTFT
    audio, _ = neucodec.generator(emb, vq=False)
    return audio
