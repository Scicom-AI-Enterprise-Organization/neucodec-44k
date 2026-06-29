# NeuCodec 44k

Scaling NeuCodec to 44.1k.

## Load from Hugging Face

The finetuned 44.1 kHz decoder weights live at
[`Scicom-intl/neucodec-44k-d20`](https://huggingface.co/Scicom-intl/neucodec-44k-d20)
(authenticate with `huggingface-cli login` or an `HF_TOKEN` env var first).

```python
import soundfile as sf
from neucodec import NeuCodec

# decoder_depth=20 MUST match the finetuned decoder (the d20 repo is a depth-20 decoder)
model = NeuCodec._from_pretrained(model_id="Scicom-intl/neucodec-44k-d20", decoder_depth=20)
model = model.eval().cuda()

# encode: any audio file (resampled to 16 kHz internally for the frozen encoder)
codes = model.encode_code("input.wav")     # [1, 1, T]  FSQ codes @ 50 tokens/sec

# decode: codes -> 44.1 kHz waveform
wav = model.decode_code(codes)              # [1, 1, T_audio]
sf.write("recon_44k.wav", wav.squeeze().cpu().numpy(), model.sample_rate)  # 44100
```

Notes:
- The encoder + FSQ codebook are unchanged from base NeuCodec — only the decoder
  was finetuned to reconstruct 44.1 kHz, so codes are interchangeable with the base
  model. `model.sample_rate == 44100`.
- `decoder_depth=20` is required: the weights are a depth-20 decoder, so loading with
  any other depth mismatches the architecture.
- The repo also ships `last.ckpt` (full PyTorch-Lightning checkpoint, with optimizer
  states) for resuming training; `pytorch_model.bin` is the weights-only inference file.
- Re-push a newer checkpoint with `python push_to_hf.py --ckpt 44k_d20/<epoch=*-step=N.ckpt> --repo Scicom-intl/neucodec-44k-d20`.

## Decoder depth & trainable parameters

Only the decoder (`generator.backbone` + `head`) and the two GAN discriminators
train; the w2v-BERT semantic encoder, acoustic encoder and the FSQ codebook path
(~637M params) stay frozen for every depth. The decoder-depth ablation (12 vs 16
vs 20 transformer layers, each finetuned to 50k batches):

| decoder depth | generator (trainable) | + discriminators (MPD+Spec) | **total trainable** | frozen | total params |
|---|---|---|---|---|---|
| 12 (base)     | 187.2M | 28.4M | **215.6M** | ~637M | ~853M |
| 16            | 237.5M | 28.4M | **265.9M** | ~637M | ~903M |
| 20            | 287.9M | 28.4M | **316.3M** | ~637M | ~953M |

Each +4 layers adds ~50M trainable params (~12.6M/layer); the discriminators
(MPD 10.28M + Spec 18.10M) are fixed. Train a given depth with
`model.codec_decoder.depth=N ckpt=extended_N.pt` (see `make_extended.py` to warm-init
the new layers by replicating the base last layer).

## Training

1. Use uv,

```bash
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
```

2. Prepare dataset,

A simple file with list of files, example, [example.txt](example.txt),

```
/path/audio1.wav
/path/audio2.wav
```

3. Run finetune,

Dry run,

```bash
python3 train.py \
log_dir=44k \
train.trainer.devices=2 \
train.trainer.max_steps=1000 \
train.trainer.min_steps=1000 \
train.trainer.val_check_interval=100 \
+train.accumulate_grad_batches=4 \
dataset.train.filelist="train.txt" \
dataset.train.batch_size=8 \
dataset.val.filelist="test.txt"
```

Check WandB at https://wandb.ai/aies-scicom-scicom-ai/neucodec_44k

Actual training,

```bash
python3 train.py \
log_dir=44k \
wandb_name=44k \
wandb_project=neucodec_44k \
every_n_train_steps=5000 \
train.trainer.devices=2 \
train.trainer.max_steps=1000000 \
train.trainer.min_steps=1000000 \
train.trainer.val_check_interval=10000 \
+train.accumulate_grad_batches=8 \
dataset.train.filelist="real_train.txt" \
dataset.train.batch_size=8 \
dataset.val.filelist="real_test.txt"
```

## TokenInterpolator — 25 TPS and 12 TPS without changing the codebook

The original NeuCodec runs at **50 TPS** (44.1kHz / hop_length 882). The `TokenInterpolator` reduces the effective token rate to **25 TPS** or **12 TPS** by skipping tokens at encode time and learning to reconstruct the missing ones at decode time.

The codebook is **completely frozen and unchanged**. No codebook retraining is required.

### How it works

```
ENCODE (25 TPS):
  audio → encode_code() → [t0, t1, t2, t3, t4, t5, ...]   50 TPS
                        → [t0,     t2,     t4,     ...]   25 TPS  (every 2nd)

DECODE (25 TPS):
  [t0, t2, t4, ...] → quantizer lookup → 25 TPS embeddings
                    → TokenInterpolator(factor=2)
                    → 50 TPS embeddings → frozen decoder → audio
```

`TokenInterpolator` repeats each low-rate embedding `factor` times, adds a learned sub-position embedding per slot, then runs a small transformer to predict the missing interleaved embeddings from context.

### Train

```bash
# 25 TPS (factor=2)
python train_interpolator.py \
  --ckpt path/to/neucodec.pt \
  --factor 2 \
  --data /path/to/wavs \
  --output interp_25tps.pt

# 12 TPS (factor=4, harder — use deeper interpolator)
python train_interpolator.py \
  --ckpt path/to/neucodec.pt \
  --factor 4 \
  --data /path/to/wavs \
  --depth 6 \
  --output interp_12tps.pt
```

Training loss is a combination of:
- **Embedding MSE** on the predicted (missing) positions vs. true 50 TPS embeddings
- **Multi-resolution mel reconstruction loss** — gradients flow through the frozen decoder back into the interpolator

Only the `TokenInterpolator` weights are updated (~few M params). NeuCodec is frozen throughout.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--factor` | `2` | `2` = 25 TPS, `4` = 12 TPS |
| `--depth` | `4` | Transformer layers in interpolator |
| `--lambda_mel` | `1.0` | Weight for mel reconstruction loss |
| `--steps` | `50000` | Training steps |
| `--batch_size` | `8` | Batch size |

### Inference

```python
from neucodec.token_interpolator import TokenInterpolator, encode_low_rate, decode_low_rate

interp = TokenInterpolator(factor=2).to(device)
interp.load_state_dict(torch.load("interp_25tps.pt"))
interp.eval()

# Encode to 25 TPS
codes_25 = encode_low_rate(neucodec, audio, factor=2)   # [B, 1, T//2]

# Decode back to 44.1kHz
audio_out = decode_low_rate(neucodec, interp, codes_25)  # [B, 1, T_audio]
```
