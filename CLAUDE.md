# CLAUDE.md — NeuCodec 44.1 kHz decoder finetune

Guidance for Claude Code (and humans) working in this repo.

## What this project does

Finetune the **decoder** of [`neuphonic/neucodec`](https://huggingface.co/neuphonic/neucodec)
so it reconstructs **44.1 kHz** audio, while leaving the encoder and the codebook
**completely unchanged**.

- **Encoder input stays 16 kHz.** The acoustic encoder (`CodecEnc`), the semantic
  encoder (w2v-BERT + `SemanticEncoder_module`), and the `fc_prior`/`fc_post_a`
  projections are **frozen**.
- **Codebook is frozen and unchanged.** The quantizer is `ResidualFSQ` (Finite
  Scalar Quantization) — it has **no learnable codebook parameters**; codes map to
  embeddings by a fixed transform. We do not retrain or change it.
- **Only the decoder trains:** `generator.backbone` (VocosBackbone) + `generator.head`
  (ISTFTHead), the part that turns code-embeddings into a waveform, now targeting
  44.1 kHz (`sample_rate=44_100`, `hop_length=882` → 50 tokens/sec). Two GAN
  discriminators (`HiFiGANMultiPeriodDiscriminator`, `SpecDiscriminator`) train
  alongside it. Losses: multi-resolution mel + STFT + feature-matching + adversarial.

> **We are NOT doing token interpolation right now.** `train_interpolator.py` /
> `neucodec/token_interpolator.py` (the 25/12 TPS path described in `README.md`) are
> **out of scope** — do not wire them into training. The task is purely "decode to
> 44.1 kHz, codebook untouched."

Training entrypoint: `train.py` (PyTorch Lightning, **manual optimization**,
gradient accumulation handled in `training_step`). Config via Hydra (`config/`).

## End-to-end signal flow (one training step)

```
wav_16k ─(frozen encoder + frozen FSQ)→ fsq_codes [B,1,T]  (no_grad)
fsq_codes ─(TRAINABLE decoder)→ gen_wav_44k
loss = mel + stft + feat_match + adversarial   vs  wav_44k (ground truth)
```

`data_module.FSDataset` produces both `wav` (16 kHz, encoder input) and `wav_24k`
(44.1 kHz target — the name is legacy) from the **same time window** of each file.

## RunPod workflow

Training runs on a **single H100 (80 GB), SECURE cloud** (COMMUNITY has no H100
stock). Storage rules, by request:
- Use **`/`** (the container disk). The scale-up run uses **1500 GB** (8 base corpora
  ~347 GB re-downloaded + `scale44k` ~192 GB + checkpoints + transient peaks). Earlier
  runs used 500 GB. `containerDiskInGb` can't be resized in place → redeploy a new pod.
- **US region** (`--country-codes US`): a far-region pod (e.g. APNIC) makes HF/dataset
  downloads slow. Kansas-City US pod measured ~47 MB/s single-stream to HF (faster with
  16 parallel workers + Xet). Always launch the data pod in the US.
- **Never `/workspace`** (that's the network volume) — `prepare_data.py` refuses it,
  and `HF_HOME`/data/venvs/checkpoints all live under `/`. We launch with `volumeInGb=0`
  so there is **no** network volume at all (the empty `/workspace` image folder is inert).

All RunPod control is via the REST API (`https://rest.runpod.io/v1`) using
`RUNPOD_API_KEY` from `.env`. Secrets (`HF_TOKEN`, `WANDB_API_KEY`, …) are in `.env`
and are rsync'd to the pod.

```bash
# 1. provision the pod (1x H100 SECURE, US region, 1500GB container disk, SSH key)
python3 runpod/launch_pod.py launch --disk-gb 1500 --country-codes US   # waits for RUNNING + prints ssh
python3 runpod/launch_pod.py status            # status + ssh endpoint
python3 runpod/launch_pod.py ssh               # print ready-to-paste ssh cmd
python3 runpod/launch_pod.py terminate --pod-id <id>   # tear it down (STOP THE BILL)

# 2. sync code, install deps, start training (reads runpod/pod.json)
./runpod/sync_and_launch.sh all                # sync + bootstrap + launch (background)
./runpod/sync_and_launch.sh tail               # follow /neucodec-44k/train.log
```

`runpod/pod.json` (gitignored) caches the pod id + SSH coords so the helper
commands work without re-passing them.

### Pod layout (everything under `/`)
| path | what |
|---|---|
| `/neucodec-44k` | the repo (rsync'd from local) |
| `/neucodec-44k/.venv` | training venv (uv, py3.12, torch 2.8 cu128) |
| `/neucodec-44k/.venv-mos` | isolated UTMOSv2 venv (py3.11) |
| `/neucodec-44k/44k` | checkpoints + `mos_eval/` + hydra logs |
| `/neucodec-44k/train.log` | training stdout/stderr |
| `/data` | extracted audio + `train.txt`/`test.txt`/`mos.txt` |
| `/hf_cache` | `HF_HOME` (downloads) |

## Data

**Log in with `HF_TOKEN`** (every download passes `token=`) and use `hf_xet`
(`HF_XET_HIGH_PERFORMANCE=1`) for fast downloads (`prepare_data.py` does this). The
current run uses **8 base corpora + the `scale44k` expansion** (336 verified ≥44 kHz
datasets). `--sources` selects which to fetch.

Base corpora (`prepare_data.py` source keys):

| source key | repo | format | size |
|---|---|---|---|
| `commonvoice` | `malaysia-ai/Multilingual-TTS` | 24× standalone `commonvoice22_sidon-*.zip` | short clips, ~100 GB |
| `sg` | `malaysia-ai/singaporean-podcast-youtube` | **split zip** `sg-podcast.zip`+`.z01..z06` | 3451 files, 1255 h |
| `malay` | `malaysia-ai/malaysian-podcast-youtube` | **split zip** `malaysian-podcast.zip`+`.z01..z11` | 19092 files, 2234 h |
| `cartoons` | `malaysia-ai/malaysian-cartoons-youtube` | standalone zips (48k subset) | ~24 GB |
| `movie` | `malaysia-ai/malaysian-movie-youtube` | standalone `part-*.zip` (48k) | ~1.8 GB |
| `expresso_read` / `expresso_conv` | `ylacombe/expresso`, `nytopop/expresso-conversational` | HF datasets-format (48k) | ~15 GB |
| `ears` | EARS (facebookresearch GitHub releases) | per-speaker zips (48k anechoic) | ~19 GB |

**`scale44k` (the "scale more dataset" source):** 336 datasets SR-verified **≥44 kHz**
(162×48k + 174×44.1k, ~192 GB), mirrored as `<name>*audio.zip` in
`malaysia-ai/Multilingual-TTS`. The zip list lives in `runpod/scale44k_zips.txt`
(derived from `…/srfilter/out/datasets_ge_44k.md`); `download_scale44k` pulls each zip,
extracts to `/data/scale44k/<stem>/`, deletes it, and drops per-zip `.done` sentinels
(resumable). Language is irrelevant — the codebook is frozen, so any clean ≥44 kHz
audio improves the decoder.

`prepare_data.py`:
- Downloads **biggest-first** (`malay,sg,commonvoice`) and **deletes archives right
  after extraction** so the transient disk peak stays well under 500 GB.
- Standalone zips → `unzip`; split zips → `7zz x` (handles multi-volume natively).
- Drops a `.done` sentinel per source → **resumable**.
- Builds filelists of `path<TAB>duration_seconds` (durations read from mp3 headers
  via `mutagen`, no decode). Holds out `mos.txt` (64) + `test.txt` (256); rest is
  `train.txt`.

```bash
/neucodec-44k/.venv/bin/python prepare_data.py --data-root /data \
    --sources malay,sg,commonvoice --mos-samples 64 --test-samples 256
```

### Data shape & the loader (important)
Extracted total ≈ **6,500 h / 3.0M files**: ~2.16M short CommonVoice mp3 clips
(~5 s, 98.9% of files) + ~22.5k long podcast files (7–22 min). `train.txt` ≈
**2.18M** lines.

`FSDataset.__getitem__` picks a random window using the precomputed duration:
- **short files (≤30 s, the vast majority):** `soundfile.read` the whole file
  in-process (~10 ms) + crop + `soxr` resample. **40× faster than spawning ffmpeg.**
- **long podcasts (>30 s):** `ffmpeg -ss <start> -t <win>` decodes only the window
  (`data_module.ffmpeg_window`).
The 16 kHz encoder input is resampled (soxr) from the same 44.1 kHz window so the
two stay aligned. `ffmpeg`, `soundfile`, `soxr` come from `bootstrap.sh`.

### CPU threads (CRITICAL on many-core pods)
The H100 pod has **224 vCPUs**. numpy/torch default each process to one thread
pool of ~224 threads; with N dataloader workers that's N×224 threads → massive
oversubscription that **starves the GPU to ~1%** (looked like a hang). Fix, applied
three ways: `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` (env in the run
script + set at the top of `train.py`/`data_module.py` before numpy/torch import) and
a DataLoader `worker_init_fn` that calls `torch.set_num_threads(1)`. With this, the
pipeline does ~9k batches/s and the GPU is the bottleneck (good). **Never remove
these on a high-core box.**

## Per-epoch MOS evaluation

At the end of **every epoch** we measure predicted naturalness MOS with **UTMOSv2**,
mirroring `…/GPUPlatform/gateway/gateway/training/tts/tts_eval.py`
(`utmosv2.create_model(pretrained=True).predict(input_path=…, num_repetitions=5)`).

Cadence: `mos.every_n_steps>0` measures MOS every N **true batches** (default 10000;
one epoch is ~45k batches at batch 48, so ~4–5 MOS points/epoch); `==0` falls back
to epoch-end. The callback fires from `on_train_batch_end` (counting true batches via
`pl_module._train_batches`, not the doubled `global_step`) or `on_train_epoch_end`.

- `mos_callback.MOSEvalCallback` (registered in `train.py` when `cfg.mos.enable`):
  on rank 0, takes `mos.txt`, runs each file through the **current** model
  (`encode_code → decode_code`, the exact reconstruction being finetuned), writes
  the reconstructed wavs (+ ground-truth refs) under `44k/mos_eval/epochNNN/`, then
  scores them.
- `mos_score.py` runs in the **isolated MOS venv** as a subprocess and prints
  `@@MOS {json}`. The callback parses it and logs to wandb:
  - `val/mos` — MOS of the decoder's reconstruction (the number to watch),
  - `val/mos_gt` — MOS of ground-truth audio (a ceiling for reference).
- MOS is **best-effort**: if the MOS venv / UTMOSv2 isn't available, the callback
  logs a warning and training continues. The faster Scicom fork is used if
  `GITHUB_TOKEN` is set, else public `sarulab-speech/UTMOSv2` (same `utmosv2` API).

Config block: `config/default.yaml → mos:` (filelist, window_sec, num_repetitions,
the MOS-venv python path, etc.).

## Training config & launch

Tunables (env) for `runpod/run_44k_finetune.sh`: `EPOCHS BATCH ACCUM NUM_WORKERS
MOS_EVERY` and the limiter `MAX_STEPS` (>0) / `MAX_TIME` ("DD:HH:MM:SS"). Limiter
priority: `MAX_TIME` > `MAX_STEPS` > `EPOCHS`. `train.trainer.devices=1`,
`limit_val_batches=0` (MOS replaces the val loop). Checkpoints every 5000 steps +
`last.ckpt`; **resumes from `44k/last.ckpt` automatically** (delete it for a clean
start). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` guards large-batch OOM.

Throughput (1×H100): batch 8 ≈ 0.37 s/step (data load 4 ms — GPU-bound). **Batch 48
fits in 63.6/81.6 GB at 100% GPU.** One epoch over 2.18M files ≈ 45k steps at batch 48;
10 epochs ≈ several days — stop early via wandb when `val/mos` plateaus.

```bash
# current run (on the pod): 10 epochs, batch 48, MOS every 20k steps
EPOCHS=10 BATCH=48 ACCUM=1 NUM_WORKERS=16 MOS_EVERY=20000 bash runpod/run_44k_finetune.sh
```

WandB: project `neucodec_44k`, run `44k`. Watch `val/mos` per epoch + the GAN/mel
losses per step.

## Checkpoints
- `44k/last.ckpt` + top-k by step. A PL checkpoint's `state_dict` keys are prefixed
  `model.` (the NeuCodec submodule). `extend_decoder.py` shows how to strip the
  prefix to get a NeuCodec-loadable `.pt`; to **resume/finetune** just point
  `train.py` at the dir (auto-detects `last.ckpt`) or pass `ckpt=<file>.pt`.
- **HF repo `Scicom-intl/neucodec-44k-d20`** holds BOTH artifacts:
  - `pytorch_model.bin` — weights only, for **inference** (`push_to_hf.py` strips the
    `model.` prefix; this is what the 12 h cron + `local_infer.py` use).
  - `last.ckpt` — the **full** PL checkpoint **with optimizer_states (×2) + lr_schedulers
    + global_step**, for **true resume**. Pull it into `<log_dir>/last.ckpt` and
    `trainer.fit(ckpt_path=…)` continues with momentum intact.
- **True resume across a redeploy:** `CKPT=null DEPTH=20` builds the depth-20 arch from
  base neucodec (extra layers init, then resume overwrites all weights), so you do **not**
  need `extended_20.pt` on a fresh pod — only `last.ckpt` in the log dir.

## Conventions / gotchas
- **Manual optimization** (`automatic_optimization=False`): gradient accumulation is
  the `train.accumulate_grad_batches` knob read inside `training_step` — **not**
  `trainer.accumulate_grad_batches` (keep that at 1, or Lightning errors under manual
  opt).
- `DataModule.get_loader` always builds `FSDataset` (the `_target_` in
  `config/dataset/default.yaml` is ignored).
- A bad/short sample returns `None` from `__getitem__`; `collate_fn` filters it.
- Editing files locally? Re-run `./runpod/sync_and_launch.sh sync` then `launch`.
- **Cost:** SECURE H100 ≈ $3.29/hr. `terminate` the pod when done.

## Key files
| file | role |
|---|---|
| `train.py` | Lightning module + trainer; registers `MOSEvalCallback` |
| `data_module.py` | `FSDataset` (ffmpeg-window loader) + `ffmpeg_window` |
| `prepare_data.py` | download/extract datasets + build filelists (on pod) |
| `mos_callback.py` | per-epoch reconstruct + score + log `val/mos` |
| `mos_score.py` | UTMOSv2 scorer (runs in `.venv-mos`) |
| `config/` | Hydra configs (`default.yaml` has the `mos:` block) |
| `runpod/launch_pod.py` | provision/status/ssh/terminate the pod (`--disk-gb`, `--country-codes`) |
| `runpod/bootstrap.sh` | system deps + uv venvs on the pod |
| `runpod/run_44k_finetune.sh` | data prep + training launch (on pod) |
| `runpod/sync_and_launch.sh` | local: rsync + bootstrap + launch |
| `runpod/redeploy_scale44k.sh` | pod-side: pull full `last.ckpt` from HF + prep all data (8 base + scale44k) + resume d20 |
| `runpod/scale44k_zips.txt` | manifest of the 336 ≥44 kHz `*audio.zip` to fetch for `scale44k` |
| `push_to_hf.py` | weights-only push to HF (inference); `local_infer.py` decodes `audio/*.mp3` |
| `neucodec/` | the NeuCodec model package (encoder/decoder/FSQ) |
| `train_interpolator.py` | OUT OF SCOPE for now (token interpolation) |
