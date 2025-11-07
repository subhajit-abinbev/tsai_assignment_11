import os, re, bz2, math, json, shutil, tarfile, sys, subprocess, unicodedata
from pathlib import Path
from typing import Iterable, Tuple

RE_MULTISPACE = re.compile(r"\s+")
RE_BAD_CHARS = re.compile(r"[^\u0980-\u09FF\s\.,!?;ঃ৳%০-৯\-–—\"'()।]")  # Keep Bengali range + punctuation
RE_MULTIDOT = re.compile(r"।{2,}")
BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def normalize_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    # Unicode normalize
    line = unicodedata.normalize("NFC", line)
    # Remove templates like {{...}}
    line = re.sub(r"\{\{.*?\}\}", " ", line)
    # Remove file/image tags
    line = re.sub(r"\[\[ফাইল:.*?\]\]", " ", line)
    # Remove URLs
    line = re.sub(r"https?://\S+", " ", line)
    # Keep Bengali + allowed punctuation
    line = RE_BAD_CHARS.sub(" ", line)
    # Collapse punctuation duplicates
    line = RE_MULTIDOT.sub("।", line)
    # Normalize digits (optionally keep Bengali digits)
    line = line.translate(BENGALI_DIGITS)
    # Collapse whitespace
    line = RE_MULTISPACE.sub(" ", line)
    return line.strip()

def iter_text_files(root: Path) -> Iterable[str]:
    for p in root.rglob("*.txt"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                norm = normalize_line(line)
                if norm:
                    yield norm

def compute_basic_metrics(raw_text_path: Path, tokenizer_encode_fn, sample_size: int = 200000) -> dict:
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    sampled = 0
    with open(raw_text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            total_chars += len(line)
            total_bytes += len(line.encode("utf-8"))
            token_ids = tokenizer_encode_fn(line)
            total_tokens += len(token_ids)
            sampled += 1
            if sampled >= sample_size:
                break
    chars_per_token = total_chars / total_tokens
    bytes_per_token = total_bytes / total_tokens
    naive_storage_bytes = total_tokens * 2  # assume <= 65535 -> 2 bytes
    compression_ratio_bytes = total_bytes / naive_storage_bytes
    return {
        "lines_sampled": sampled,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "chars_per_token": chars_per_token,
        "bytes_per_token": bytes_per_token,
        "assignment_compression_ratio": chars_per_token,  # using this as ratio ≥ 3 requirement
        "approx_byte_compression_ratio": compression_ratio_bytes
    }

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)