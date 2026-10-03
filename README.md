# 🖼️ Multimodal Image Captioning — ClipCap-style Prefix Tuning

A lightweight, locally-runnable image captioning system that bridges a **frozen vision encoder** to a **frozen language model** through a small trainable projector.

## Architecture

```
image
  → I-JEPA ViT-H/14 (frozen, 630M params)
  → mean-pool patch tokens → 1280-dim feature vector
  → Projector MLP (trainable, ~2M params)
      Linear(1280 → 256) → GELU → Linear(256 → 8×768)
  → 8 pseudo-token embeddings (768-dim each)
  → GPT-2 small (frozen, 124M params)
  → autoregressive caption
```

This is a **ClipCap-style prefix-tuning** setup: only the Projector MLP is trained. GPT-2 treats the projected image features as a "prefix" context and generates captions conditioned on them.

## Target Hardware

- **Apple M-series MacBook** (16 GB unified memory)
- Uses PyTorch's **MPS backend** automatically (falls back to CPU)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python download_data.py
```

This pulls **Flickr8k** (8,000 images, 5 captions each) via `kagglehub` and symlinks it into `data/Images/` and `data/captions.txt`.

> **Note:** You need a Kaggle account. Set up your credentials via `kagglehub` or the `~/.kaggle/kaggle.json` file.

### 3. Extract features (one-time, slow step)

```bash
python extract_features.py
```

This runs the frozen I-JEPA ViT-H/14 encoder over all 8,000 images and caches the mean-pooled features to `features.pt`. **This only needs to run once** — all subsequent training loads from cache.

- **ViT-Huge:** ~10–15 min on MPS, produces `features.pt` (~20 MB)
- If too slow, use the ViT-Base fallback:
  ```bash
  python extract_features.py --fallback
  ```

### 4. Train the projector

```bash
python train.py
```

Trains the Projector MLP (only ~2M trainable params) against frozen GPT-2 for 8 epochs. Saves the best checkpoint to `checkpoints/best.pt`.

**Optional: fine-tune GPT-2 as well**

```bash
python train.py --unfreeze_gpt2
```

This adds GPT-2's parameters to the optimizer at a lower learning rate (2e-5 vs 1e-4 for the projector).

### 5. Generate a caption

```bash
python generate.py --image path/to/your/photo.jpg
```

Or use the fallback encoder:

```bash
python generate.py --image photo.jpg --fallback
```

### 6. Interactive demo

```bash
python demo.py
```

Opens a Gradio interface at `http://localhost:7860`. Upload any image and get a caption.

## File Structure

```
├── download_data.py      # Step 1: Download Flickr8k dataset
├── dataset.py             # Caption parsing, train/val splitting
├── extract_features.py    # Step 2: One-time feature extraction (slow)
├── model.py               # Projector MLP + CaptionModel (GPT-2 wrapper)
├── train.py               # Step 3: Training loop
├── generate.py            # Step 4: Single-image inference (CLI)
├── demo.py                # Step 5: Gradio web interface
├── retrieval_demo.py      # Side demo: visual similarity retrieval
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/                  # Created by download_data.py
│   ├── Images/*.jpg
│   └── captions.txt
├── features.pt            # Created by extract_features.py
└── checkpoints/
    └── best.pt            # Created by train.py
```

## Visual Retrieval Demo (Side Demo)

A standalone script that proves **I-JEPA's embeddings capture meaningful visual similarity** — with zero text or language involved. No GPT-2, no captions, no training needed. It only uses the pre-cached `features.pt` vectors.

### What it does

1. Loads the `features.pt` cache (~8000 × 1280 float16 vectors)
2. Picks a query image (user-specified or random)
3. Computes cosine similarity against every other image in a single vectorized matrix operation
4. Prints a ranked table of nearest neighbours with similarity scores
5. Saves a visual grid (`retrieval_result.png`): query on the left, top-K neighbours to the right, each labelled with its score

### Usage

```bash
# Specific query image
python retrieval_demo.py --query 1000268201_693b08cb0e.jpg --top_k 5

# Random query (great for quick demos)
python retrieval_demo.py --top_k 8

# All options
python retrieval_demo.py --features features.pt --images_dir data/Images --query <filename> --top_k 5 --output retrieval_result.png
```

### Output

Terminal output — a ranked similarity table:
```
🔍  Query image: 667626_18933d713e.jpg

Top-5 nearest neighbours (cosine similarity):

   Rank   Filename                                 Similarity
   ────── ──────────────────────────────────────── ──────────
   1      3637013_c675de7705.jpg                       0.9312
   2      2923034235_e30ed tried.jpg                   0.9108
   …
```

Image output — a labelled grid saved to `retrieval_result.png`:

| QUERY | #1 sim=0.93 | #2 sim=0.91 | #3 sim=0.89 | #4 sim=0.87 | #5 sim=0.85 |
|-------|-------------|-------------|-------------|-------------|-------------|

> **Why this matters for the report:** This demonstrates that I-JEPA's self-supervised pre-training (no labels, no text) produces embeddings where visually/semantically similar images are close in cosine space. The captioning pipeline builds on top of these already-meaningful representations.

### Prerequisites

- `features.pt` must exist (run `python extract_features.py` first)
- `data/Images/` must contain the original Flickr8k images
- `matplotlib` must be installed (`pip install matplotlib`, or it's already in `requirements.txt`)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pre-extract features to disk | The vision encoder is huge and frozen — running it once and caching avoids repeated ~10 min waits |
| Mean-pool patch tokens | Simplest possible aggregation; more complex approaches (attention pooling, learned queries) are out of scope |
| 8 pseudo-tokens | Enough to carry meaningful visual context; more tokens = more memory per sample |
| Projector hidden dim = 256 | Keeps param count small (~2M); this is a demo, not a production system |
| Labels masked at prefix positions | Standard practice — the model should only be penalized for predicting caption tokens |
| Beam search + no_repeat_ngram | Reduces repetitive outputs without needing nucleus sampling tuning |

## Vision Encoder Fallback

The primary encoder is **`facebook/ijepa_vith14_1k`** — a ViT-Huge with 630M params and 1280-dim features. If this is too slow or uses too much memory:

```bash
# Use ViT-Base (86M params, 768-dim) instead — much faster
python extract_features.py --fallback
python generate.py --image photo.jpg --fallback
python demo.py --fallback
```

When using the fallback, features are zero-padded from 768→1280 so the downstream Projector MLP doesn't need to change. **Note:** You must use the same encoder for feature extraction and inference.

## Hyperparameters

| Parameter | Default | Flag |
|-----------|---------|------|
| Epochs | 8 | `--epochs` |
| Batch size | 16 | `--batch_size` |
| Projector LR | 1e-4 | `--lr` |
| GPT-2 LR | 2e-5 | `--gpt2_lr` |
| Max caption length | 64 tokens | `--max_length` |
| Weight decay | 0.01 | `--weight_decay` |

## Expected Timeline

| Step | Time (MPS, M-series) |
|------|---------------------|
| Download dataset | ~2 min |
| Extract features (ViT-H) | ~10–15 min |
| Extract features (ViT-Base fallback) | ~2–3 min |
| Train 8 epochs | ~5–10 min |
| Generate one caption | ~5–10 sec |

**Total from zero to first caption: ~20–30 min** (after one-time setup).

## Limitations

This is a **demo project** to prove the architecture works end-to-end:

- Caption quality is limited by the small dataset (8K images) and frozen GPT-2
- No cross-attention, no RAG, no sophisticated decoding — just prefix-tuning
- The projector is intentionally small; scaling it up would improve quality
- Flickr8k captions are short and formulaic ("A man in a red shirt…")

For production quality, consider: larger datasets (COCO, CC3M), unfreezing GPT-2, cross-attention layers, and better decoding strategies.
