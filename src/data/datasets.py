"""
Used in the OG paper: https://arxiv.org/abs/1310.4546
Source: http://mattmahoney.net/dc/text8.zip
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent.parent / "data"
TEXT8 = "http://mattmahoney.net/dc/text8.zip"
UNK = "<UNK>"


def download_text8(ddir=None):
    if ddir is None:
        ddir = DATA

    ddir.mkdir(parents=True, exist_ok=True)
    text8 = ddir / "text8"
    zip_path = ddir / "text8.zip"
    if text8.exists():
        return text8

    urllib.request.urlretrieve(TEXT8, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ddir)
    zip_path.unlink()

    return text8


def load_text8(ddir=None):
    text8_path = download_text8(ddir)
    with text8_path.open(encoding="utf-8") as f:
        text = f.read()
    return text.split()
