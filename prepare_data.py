#!/usr/bin/env python3
"""Download + extract the training audio and build the train/test/MOS filelists.

Runs ON THE POD. Pulls three Malaysia-AI datasets straight onto the container
disk ("/", never the /workspace network volume), extracts them, then writes
filelists of `path<TAB>duration_seconds` lines that data_module.FSDataset reads.

Sources (downloaded BIGGEST-FIRST so the transient archive+extract disk peak
stays low):

  malaysia-ai/malaysian-podcast-youtube     split zip  (malaysian-podcast.zip + .z01..z11)
  malaysia-ai/singaporean-podcast-youtube   split zip  (sg-podcast.zip + .z01..z06)
  malaysia-ai/Multilingual-TTS              24x standalone commonvoice22_sidon-*.zip

Fast + authenticated downloads: we log in with HF_TOKEN and enable hf_transfer
+ hf_xet. Archives are deleted immediately after extraction to bound disk use.
Each source drops a `.done` sentinel so a re-run is resumable.

Audio is mp3 (CommonVoice short clips + long-form podcasts). Durations are read
from headers (mutagen) without decoding; the loader uses them to pick a random
window via ffmpeg, so long podcast files never get fully decoded.

Usage:
  python prepare_data.py --data-root /data \
      [--sources malay,sg,commonvoice] [--mos-samples 64] [--test-samples 256]
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Speed knobs must be set before huggingface_hub is imported. The datasets are
# Xet-backed, so HF_XET_HIGH_PERFORMANCE is the one that matters now.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".opus", ".ogg")

# name -> (repo_id, kind, [download globs], main_archive_for_split)
SOURCES = {
    "malay": (
        "malaysia-ai/malaysian-podcast-youtube", "split",
        ["malaysian-podcast.z*", "malaysian-podcast.zip"], "malaysian-podcast.zip",
    ),
    "sg": (
        "malaysia-ai/singaporean-podcast-youtube", "split",
        ["sg-podcast.z*", "sg-podcast.zip"], "sg-podcast.zip",
    ),
    "commonvoice": (
        "malaysia-ai/Multilingual-TTS", "standalone",
        ["commonvoice22_sidon-*.zip"], None,
    ),
}
DEFAULT_ORDER = ["malay", "sg", "commonvoice"]  # biggest-first

# HF datasets-format audio (real human studio speech @ 48k, not synthetic TTS).
# name -> (repo_id, config, split). Exported to /data/<name>/*.wav.
HF_DATASETS = {
    "expresso_read": ("ylacombe/expresso", "read", "train"),            # 11h expressive read
    "expresso_conv": ("nytopop/expresso-conversational", "conversational", "train"),  # 30h improvised dialogue
}


def log(m: str) -> None:
    print(m, flush=True)


def disk_free_gb(path: str) -> float:
    # Walk up to the first existing ancestor so a not-yet-created dir is fine.
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p or "/").free / 1e9


def hf_login() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False)
            log("[hf] logged in with HF_TOKEN")
        except Exception as e:  # noqa: BLE001
            log(f"[hf] login warning: {e}")
    else:
        log("[hf] no HF_TOKEN set — downloading anonymously (slower / may rate-limit)")
    return tok


def find_7z() -> str:
    for cand in ("7zz", "7z", "7za"):
        if shutil.which(cand):
            return cand
    sys.exit("no 7-Zip binary found (need 7zz/7z) — run bootstrap.sh first")


def download(repo: str, patterns: list[str], dest: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    log(f"[dl] {repo}  patterns={patterns} -> {dest}")
    snapshot_download(
        repo_id=repo, repo_type="dataset", allow_patterns=patterns,
        local_dir=str(dest), token=token, max_workers=16,
    )


def extract_standalone(archive_dir: Path, out_root: Path) -> None:
    """Each CommonVoice zip carries its own top folder; unzip into out_root then
    delete the zip to keep disk flat."""
    out_root.mkdir(parents=True, exist_ok=True)
    zips = sorted(archive_dir.glob("commonvoice22_sidon-*.zip"))
    for i, z in enumerate(zips, 1):
        log(f"[unzip] ({i}/{len(zips)}) {z.name}  free={disk_free_gb(str(out_root)):.0f}GB")
        rc = subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(out_root)]).returncode
        if rc != 0:
            log(f"[unzip] WARNING rc={rc} on {z.name}")
        z.unlink(missing_ok=True)


def extract_split(archive_dir: Path, main_archive: str, out_root: Path) -> None:
    """7-Zip handles the multi-volume zip natively when all .z* parts sit next to
    the main .zip. Extract, then delete every part."""
    sevenz = find_7z()
    main = archive_dir / main_archive
    out_root.mkdir(parents=True, exist_ok=True)
    log(f"[7z] {sevenz} x {main.name} -> {out_root}  free={disk_free_gb(str(out_root)):.0f}GB")
    rc = subprocess.run([sevenz, "x", str(main), f"-o{out_root}", "-y", "-mmt8"]).returncode
    if rc != 0:
        sys.exit(f"[7z] extraction failed rc={rc} for {main}")
    for part in archive_dir.glob(main_archive.split(".")[0] + ".z*"):
        part.unlink(missing_ok=True)
    (archive_dir / main_archive).unlink(missing_ok=True)


def export_hf_dataset(name: str, data_root: Path, token: str | None) -> None:
    """Stream an HF datasets-format audio dataset and dump each clip as a wav
    under /data/<name>/ so it merges into the same filelist as the zip sources."""
    import soundfile as sf
    repo, config, split = HF_DATASETS[name]
    out_root = data_root / name
    done = out_root / ".done"
    if done.exists():
        log(f"[skip] {name} already exported ({out_root})")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    log(f"[hf-ds] streaming {repo}:{config}:{split} -> {out_root}")
    ds = load_dataset(repo, config, split=split, streaming=True, token=token)
    n = 0
    for i, row in enumerate(ds):
        a = row.get("audio")
        if not a or a.get("array") is None:
            continue
        try:
            sf.write(str(out_root / f"{i:06d}.wav"), a["array"], int(a["sampling_rate"]), subtype="PCM_16")
            n += 1
        except Exception as e:  # noqa: BLE001
            log(f"[hf-ds] skip row {i}: {e}")
        if n and n % 2000 == 0:
            log(f"[hf-ds]   {name}: {n} files")
    done.write_text("ok\n")
    log(f"[done] {name}: {n} wavs -> {out_root}  free={disk_free_gb(str(data_root)):.0f}GB")


def process_source(name: str, data_root: Path, token: str | None) -> None:
    repo, kind, patterns, main = SOURCES[name]
    out_root = data_root / name
    done = out_root / ".done"
    if done.exists():
        log(f"[skip] {name} already extracted ({out_root})")
        return
    archive_dir = data_root / "_archives" / name
    download(repo, patterns, archive_dir, token)
    if kind == "standalone":
        extract_standalone(archive_dir, out_root)
    else:
        extract_split(archive_dir, main, out_root)
    shutil.rmtree(archive_dir, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    done.write_text("ok\n")
    log(f"[done] {name}: extracted to {out_root}  free={disk_free_gb(str(data_root)):.0f}GB")


# --------------------------------------------------------------------------- #
# Durations + filelists
# --------------------------------------------------------------------------- #
def probe_duration(path: str) -> tuple[str, float]:
    """Header-only duration (seconds). mutagen for mp3 (no decode); soundfile
    fallback. Returns 0.0 if unreadable."""
    try:
        if path.lower().endswith(".mp3"):
            from mutagen.mp3 import MP3
            return path, float(MP3(path).info.length)
        import soundfile as sf
        info = sf.info(path)
        return path, float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return path, 0.0


def build_filelists(data_root: Path, mos_n: int, test_n: int, min_dur: float, seed: int) -> None:
    log("[scan] globbing audio files …")
    files: list[str] = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(str(data_root / "**" / f"*{ext}"), recursive=True))
    files = sorted(set(files))
    log(f"[scan] {len(files)} audio files found; probing durations …")

    rows: list[tuple[str, float]] = []
    with ProcessPoolExecutor(max_workers=min(32, (os.cpu_count() or 8))) as ex:
        for i, (p, d) in enumerate(ex.map(probe_duration, files, chunksize=256)):
            if d >= min_dur:
                rows.append((p, d))
            if (i + 1) % 50000 == 0:
                log(f"[scan]   probed {i + 1}/{len(files)}")
    log(f"[scan] {len(rows)} files >= {min_dur}s ({sum(d for _, d in rows) / 3600:.1f}h total)")

    rng = random.Random(seed)
    rng.shuffle(rows)
    mos = rows[:mos_n]
    test = rows[mos_n:mos_n + test_n]
    train = rows[mos_n + test_n:]

    def write(name: str, items: list[tuple[str, float]]) -> None:
        out = data_root / name
        with open(out, "w") as f:
            for p, d in items:
                f.write(f"{p}\t{d:.3f}\n")
        log(f"[write] {out}  ({len(items)} files)")

    write("train.txt", train)
    write("test.txt", test)
    write("mos.txt", mos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--sources", default=",".join(DEFAULT_ORDER),
                    help="comma list of: malay,sg,commonvoice")
    ap.add_argument("--mos-samples", type=int, default=64)
    ap.add_argument("--test-samples", type=int, default=256)
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--skip-download", action="store_true",
                    help="only (re)build filelists from already-extracted audio")
    a = ap.parse_args()

    data_root = Path(a.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    if "/workspace" in str(data_root.resolve()):
        sys.exit("refusing to use /workspace (network volume) — pass --data-root under /")

    token = hf_login()
    if not a.skip_download:
        for name in [s.strip() for s in a.sources.split(",") if s.strip()]:
            log(f"\n===== source: {name}  (free={disk_free_gb(str(data_root)):.0f}GB) =====")
            if name in SOURCES:
                process_source(name, data_root, token)
            elif name in HF_DATASETS:
                export_hf_dataset(name, data_root, token)
            else:
                sys.exit(f"unknown source {name!r}; choose from {list(SOURCES) + list(HF_DATASETS)}")

    build_filelists(data_root, a.mos_samples, a.test_samples, a.min_duration, a.seed)
    log("\n[prepare_data] complete.")


if __name__ == "__main__":
    main()
