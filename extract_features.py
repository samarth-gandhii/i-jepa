#!/usr/bin/env python3
"""
extract_features.py — Run the frozen vision encoder ONCE over every image
and cache the resulting feature vectors to disk.

Output: features.pt  (dict: filename → float16 tensor of shape [1280])

This is the slow step in the pipeline. Everything downstream loads the
cache, so you only need to run this once.

Encoder hierarchy:
  1. facebook/ijepa_vith14_1k  (ViT-Huge/14, ~630M params, dim 1280)
  2. google/vit-base-patch16-224 (ViT-Base, ~86M params, dim 768)   [fallback]

The fallback is auto-selected if the primary model fails to load, or you
can force it with --fallback.  When using the fallback, features are
projected up to 1280-dim with a zero-padded vector so the downstream
Projector MLP doesn't need to change.
"""

import argparse
import os
import sys
from typing import Dict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor, ViTModel, ViTImageProcessor

from dataset import load_captions


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
PRIMARY_MODEL = "facebook/ijepa_vith14_1k"
FALLBACK_MODEL = "google/vit-base-patch16-224"
FEATURE_DIM = 1280  # ViT-H/14 hidden size — everything downstream expects this


def load_encoder(use_fallback: bool = False):
    """
    Returns (model, processor, actual_dim, model_name).

    If use_fallback is True, or the primary model fails to load, we fall
    back to ViT-Base (dim 768).
    """
    if not use_fallback:
        try:
            print(f"Loading primary encoder: {PRIMARY_MODEL} …")
            processor = AutoImageProcessor.from_pretrained(PRIMARY_MODEL)
            model = AutoModel.from_pretrained(PRIMARY_MODEL)
            dim = model.config.hidden_size  # 1280 for ViT-H/14
            print(f"   ✅  Loaded {PRIMARY_MODEL}  (hidden_size={dim})")
            return model, processor, dim, PRIMARY_MODEL
        except Exception as e:
            print(f"   ⚠️  Could not load {PRIMARY_MODEL}: {e}")
            print("   Falling back to ViT-Base …")

    print(f"Loading fallback encoder: {FALLBACK_MODEL} …")
    processor = ViTImageProcessor.from_pretrained(FALLBACK_MODEL)
    model = ViTModel.from_pretrained(FALLBACK_MODEL)
    dim = model.config.hidden_size  # 768 for ViT-Base
    print(f"   ✅  Loaded {FALLBACK_MODEL}  (hidden_size={dim})")
    return model, processor, dim, FALLBACK_MODEL


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(
    images_dir: str,
    captions_path: str,
    output_path: str,
    use_fallback: bool = False,
    batch_size: int = 8,
):
    """
    Extract mean-pooled patch features for every image referenced in
    captions.txt and save them as a dict[filename → float16 tensor].
    """
    device = get_device()
    print(f"Device: {device}")

    # Load encoder
    model, processor, actual_dim, model_name = load_encoder(use_fallback)
    model = model.to(device).eval()

    # Gather image filenames from captions
    captions_dict = load_captions(captions_path)
    filenames = sorted(captions_dict.keys())
    print(f"Extracting features for {len(filenames)} images …")

    features: Dict[str, torch.Tensor] = {}
    skipped = 0

    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(filenames), batch_size), desc="Batches"):
        batch_names = filenames[batch_start : batch_start + batch_size]
        batch_images = []
        valid_names = []

        for fname in batch_names:
            img_path = os.path.join(images_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                valid_names.append(fname)
            except Exception as e:
                print(f"   ⚠️  Skipping {fname}: {e}")
                skipped += 1

        if not batch_images:
            continue

        # Preprocess and run encoder
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool patch tokens (last_hidden_state excludes CLS if model
        # provides it separately; we pool everything for simplicity)
        # Shape: [B, num_patches, hidden_dim]
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)  # [B, hidden_dim]

        # If using fallback (dim 768), zero-pad to 1280
        if actual_dim < FEATURE_DIM:
            pad = torch.zeros(
                pooled.shape[0],
                FEATURE_DIM - actual_dim,
                device=pooled.device,
            )
            pooled = torch.cat([pooled, pad], dim=1)

        # Store as float16 to save disk
        pooled = pooled.cpu().half()
        for name, vec in zip(valid_names, pooled):
            features[name] = vec

    print(f"\nExtracted {len(features)} feature vectors  (skipped {skipped} images)")
    print(f"Feature shape: {FEATURE_DIM}")

    # Save
    torch.save(features, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"💾  Saved to {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache vision encoder features for all Flickr8k images."
    )
    parser.add_argument(
        "--images_dir",
        default=os.path.join(os.path.dirname(__file__), "data", "Images"),
        help="Path to the directory of .jpg images.",
    )
    parser.add_argument(
        "--captions_path",
        default=os.path.join(os.path.dirname(__file__), "data", "captions.txt"),
        help="Path to captions.txt.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "features.pt"),
        help="Where to save the feature cache.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use google/vit-base-patch16-224 instead of I-JEPA ViT-H/14.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Images per batch during extraction.",
    )
    args = parser.parse_args()

    extract_features(
        images_dir=args.images_dir,
        captions_path=args.captions_path,
        output_path=args.output,
        use_fallback=args.fallback,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
