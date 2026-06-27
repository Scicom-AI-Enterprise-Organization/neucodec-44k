#!/usr/bin/env python3
"""Probe the sample rate of HF zip-based audio datasets WITHOUT downloading the
whole zip: range-read the zip's central directory, then read just the header of
the smallest audio entry. Run where the network to HF is reliable (the pod)."""
import io, os, sys, zipfile, urllib.request, traceback
import soundfile as sf
from huggingface_hub import hf_hub_url

TOK = os.environ["HF_TOKEN"]

class R(io.RawIOBase):
    def __init__(self, u):
        self.pos = 0
        r = urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers={"Authorization": f"Bearer {TOK}"}))
        self.size = int(r.headers.get("Content-Length") or r.headers.get("x-linked-size") or 0)
        self.u = r.geturl()
    def seek(self, o, w=0): self.pos = o if w == 0 else (self.pos + o if w == 1 else self.size + o); return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def read(self, n=-1):
        if n == -1: n = self.size - self.pos
        return urllib.request.urlopen(urllib.request.Request(self.u, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})).read()

# (repo, a single standalone zip file in it)
TARGETS = [
    ("malaysia-ai/malaysian-movie-youtube", "part-0-0.zip"),
    ("malaysia-ai/malaysian-dialects-youtube", "part-0-1.zip"),
    ("malaysia-ai/malay-classic-youtube", "part-1-0.zip"),
    ("malaysia-ai/tamil-youtube", "part-0-1.zip"),
]

for repo, fn in TARGETS:
    try:
        zf = zipfile.ZipFile(R(hf_hub_url(repo, fn, repo_type="dataset")))
        aud = [i for i in zf.infolist() if i.filename.lower().endswith((".mp3", ".wav", ".flac"))]
        e = min(aud, key=lambda i: i.file_size)
        with zf.open(e) as f:
            info = sf.info(io.BytesIO(f.read(400000)))
        print(f">>> {repo.split('/')[1]}: SR={info.samplerate} Hz ch={info.channels} entries={len(aud)}", flush=True)
    except Exception:
        print(f"!! {repo.split('/')[1]}:", flush=True)
        traceback.print_exc()
