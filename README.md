# Bengali BPE Tokenizer Project

## Goals
- Train a BPE tokenizer (vocab > 5000) on Bengali Wikipedia dump.
- Achieve compression ratio >= 3 (defined as characters per token, plus alternative metrics).
- Upload tokenizer to Hugging Face Hub.
- Deploy interactive Space for tokenization demos.

## Components
1. `training_notebook.ipynb`: End-to-end workflow.
2. `src/custom_bpe.py`: Educational custom BPE implementation.
3. `src/utils.py`: Helper functions (download, cleaning, metrics).
4. `space/app.py`: Gradio app for Hugging Face Space.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
```

## Run Notebook
```bash
jupyter lab
# Open training_notebook.ipynb
```

## Train & Upload
Inside the notebook you'll:
- Download: Bengali Wikipedia dump (`bnwiki-latest-pages-articles.xml.bz2`)
- Extract with WikiExtractor
- Normalize & clean
- Train:
  - Custom BPE (educational, slower)
  - Hugging Face `tokenizers` BPE (fast)
- Evaluate compression metrics
- Push to Hugging Face:
  - Create repo: `huggingface-cli repo create bengali-bpe-tokenizer`
  - Use notebook cell to push artifacts

## Compression Metrics
We compute:
- `chars_per_token = total_characters / total_tokens`
- `bytes_per_token = total_bytes / total_tokens`
- `ratio_variant = total_characters / total_tokens` (used as "compression ratio" for assignment target ≥ 3)
- Optional: serialization size vs raw text size (approx).

### Latest Training Snapshot (2025-11-07)
- **Training subset:** first 5000 lines of `clean_corpus.txt`
- **Tokenizer config:** `vocab_size=16000`, `min_freq=3`, `dropout=0.05`, normalization via `normalize_bengali(map_digits=True, keep_latin=True)`
- **Lines sampled for metrics:** 5000
- **Results:**
  - `chars_per_token` = 4.5499
  - `bytes_per_token` = 11.6014
  - `assignment_compression_ratio` = 4.5499
  - `approx_byte_compression_ratio` = 5.8007
  - `total_tokens` = 5,811,945 | `total_chars` = 26,444,005 | `total_bytes` = 67,426,621

These figures meet the ≥3 compression ratio requirement and reflect the current Hugging Face deployment artifacts.

## Hugging Face Space
`space/app.py` loads the tokenizer (once uploaded) and exposes:
- Encode text
- Decode tokens
- Show tokens, IDs, lengths, average chars per token

## Reproducibility
- Set random seeds
- Deterministic merges
- Log merges & vocab coverage stats

## License
Wikipedia content: CC BY-SA 3.0  
Code: MIT (adjust if needed)

## Next Steps
- Optionally add Bengali news + books for broader coverage
- Add script to benchmark vs SentencePiece
- Add perplexity-like proxy (subword fragmentation rate)

## Contact
Author: (Your Name / GitHub handle)