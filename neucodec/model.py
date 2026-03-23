from typing import Optional, Dict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
from huggingface_hub import PyTorchModelHubMixin, ModelHubMixin, hf_hub_download
from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2BertModel

from .codec_encoder import CodecEncoder
from .codec_encoder_distill import DistillCodecEncoder
from .codec_decoder_vocos import CodecDecoderVocos
from .module import SemanticEncoder


class NeuCodec(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/neuphonic/neucodec",
    license="apache-2.0",
):

    def __init__(self, sample_rate: int, hop_length: int, decoder_depth: int = 12):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0", output_hidden_states=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.CodecEnc = CodecEncoder()
        self.generator = CodecDecoderVocos(hop_length=hop_length, depth=decoder_depth)
        self.fc_prior = nn.Linear(2048, 2048)
        self.fc_post_a = nn.Linear(2048, 1024)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = False,
        local_ckpt_path: str = None,
        **model_kwargs,
    ):
        if model_id == "neuphonic/neucodec":
            ignore_keys = ["fc_post_s", "SemanticDecoder"]
        elif model_id == "neuphonic/distill-neucodec":
            ignore_keys = []
        else:
            ignore_keys = []

        if model_id is not None:
            ckpt_path = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
            )
        else:
            # incase we interpolate the weight to become 960 instead train from scratch 
            ckpt_path = local_ckpt_path

        # initialize model
        decoder_depth = model_kwargs.pop('decoder_depth', 12)
        model = cls(48_000, 960, decoder_depth=decoder_depth)

        # load weights
        state_dict = torch.load(ckpt_path, map_location)
        contains_list = lambda s, l: any(i in s for i in l)
        state_dict = {
            k:v for k, v in state_dict.items() 
            if not contains_list(k, ignore_keys)
        }

        # Filter out keys with shape mismatches (e.g. 48k model vs 24k checkpoint)
        model_state = model.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        model.load_state_dict(state_dict, strict=False)

        return model
    
    def _prepare_audio(self, audio_or_path: torch.Tensor | Path | str):
        
        # load from file
        if isinstance(audio_or_path, (Path, str)):
            y, sr = torchaudio.load(audio_or_path)
            if sr != 16_000:
                y, sr = (T.Resample(sr, 16_000)(y), 16_000)
                y = y[None, :]  # [1, T] -> [B, 1, T]

        # ensure input tensor is of correct shape
        elif isinstance(audio_or_path, torch.Tensor):
            y = audio_or_path
            if len(y.shape) == 3:
                y = audio_or_path
            else:
                raise ValueError(
                    f"NeuCodec expects tensor audio input to be of shape [B, 1, T] -- received shape: {y.shape}"
                )

        # pad audio
        pad_for_wav = 320 - (y.shape[-1] % 320)
        y = torch.nn.functional.pad(y, (0, pad_for_wav))
        
        return y
        
    def encode_code(self, audio_or_path: torch.Tensor | Path | str) -> torch.Tensor:
        """
        Args:
            audio_or_path: torch.Tensor [B, 1, T] | Path | str, input audio

        Returns:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes
        """
         
        # prepare inputs
        y = self._prepare_audio(audio_or_path)
        semantic_features = self.feature_extractor(
            [w for w in y.squeeze(1).cpu()], sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(self.device)

        # acoustic encoding
        acoustic_emb = self.CodecEnc(y.to(self.device))
        acoustic_emb = acoustic_emb.transpose(1, 2)

        # semantic encoding
        semantic_output = (
            self.semantic_model(semantic_features).hidden_states[16].transpose(1, 2)
        )
        semantic_encoded = self.SemanticEncoder_module(semantic_output)

        # concatenate embeddings
        if acoustic_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
            acoustic_emb = acoustic_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]        
        concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # quantize
        _, fsq_codes, _ = self.generator(concat_emb, vq=True)
        return fsq_codes

    def encode_code_from_features(self, audio: torch.Tensor, semantic_features: torch.Tensor) -> torch.Tensor:
        """Encode using pre-computed semantic features, avoiding CPU feature extraction.

        Args:
            audio: torch.Tensor [B, 1, T], 16kHz input audio
            semantic_features: torch.Tensor [B, seq_len, feat_dim], pre-computed features

        Returns:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes
        """
        y = self._prepare_audio(audio)
        semantic_features = semantic_features.to(self.device)

        # acoustic encoding
        acoustic_emb = self.CodecEnc(y.to(self.device))
        acoustic_emb = acoustic_emb.transpose(1, 2)

        # semantic encoding
        semantic_output = (
            self.semantic_model(semantic_features).hidden_states[16].transpose(1, 2)
        )
        semantic_encoded = self.SemanticEncoder_module(semantic_output)

        # concatenate embeddings
        if acoustic_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
            acoustic_emb = acoustic_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]
        concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # quantize
        _, fsq_codes, _ = self.generator(concat_emb, vq=True)
        return fsq_codes

    def decode_code(self, fsq_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes

        Returns:
            recon: torch.Tensor [B, 1, T], reconstructed 48kHz audio
        """

        fsq_post_emb = self.generator.quantizer.get_output_from_indices(fsq_codes.transpose(1, 2))
        fsq_post_emb = fsq_post_emb.transpose(1, 2)
        fsq_post_emb = self.fc_post_a(fsq_post_emb.transpose(1, 2)).transpose(1, 2) 
        recon = self.generator(fsq_post_emb.transpose(1, 2), vq=False)[0]
        return recon
