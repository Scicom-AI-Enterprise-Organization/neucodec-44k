# NeuCodec 44k

Scaling NeuCodec to 44.1k.

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

Check WanDB at https://wandb.ai/huseinzol05/wandb_project

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
