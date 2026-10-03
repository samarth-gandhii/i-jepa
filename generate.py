#!/usr/bin/env python3
"""
generate.py — Generate a caption for a single image using a trained checkpoint.

This script runs the vision encoder on the fly (no pre-cached features
needed), projects the features through the trained Projector, and calls
GPT-2's generate() to produce a caption.

Usage:
    python generate.py --image path/to/photo.jpg
    python generate.py --image path/to/photo.jpg --fallback   # use ViT-Base encoder
"""

import argparse
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, ViTModel, ViTImageProcessor

from model import ProjectorMLP, CaptionModel


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Encoder loading (same logic as extract_features.py)
# ---------------------------------------------------------------------------
PRIMARY_MODEL = "facebook/ijepa_vith14_1k"
FALLBACK_MODEL = "google/vit-base-patch16-224"
FEATURE_DIM = 1280


def load_encoder(use_fallback: bool = False):
    """Returns (model, processor, actual_dim)."""
    if not use_fallback:
        try:
            print(f"Loading encoder: {PRIMARY_MODEL} …")
            processor = AutoImageProcessor.from_pretrained(PRIMARY_MODEL)
            model = AutoModel.from_pretrained(PRIMARY_MODEL)
            dim = model.config.hidden_size
            print(f"   ✅  Loaded  (dim={dim})")
            return model, processor, dim
        except Exception as e:
            print(f"   ⚠️  Could not load primary model: {e}")
            print("   Falling back to ViT-Base …")

    print(f"Loading fallback encoder: {FALLBACK_MODEL} …")
    processor = ViTImageProcessor.from_pretrained(FALLBACK_MODEL)
    model = ViTModel.from_pretrained(FALLBACK_MODEL)
    dim = model.config.hidden_size
    print(f"   ✅  Loaded  (dim={dim})")
    return model, processor, dim


# ---------------------------------------------------------------------------
# Feature extraction for a single image
# ---------------------------------------------------------------------------
def extract_single_feature(
    image_path: str,
    encoder,
    processor,
    actual_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the encoder on one image and return a [1, 1280] feature tensor.
    """
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    encoder.eval()
    with torch.no_grad():
        outputs = encoder(**inputs)

    hidden = outputs.last_hidden_state  # [1, N, dim]
    pooled = hidden.mean(dim=1)         # [1, dim]

    # Pad to 1280 if using fallback
    if actual_dim < FEATURE_DIM:
        pad = torch.zeros(1, FEATURE_DIM - actual_dim, device=device)
        pooled = torch.cat([pooled, pad], dim=1)

    return pooled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a caption for a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt"),
        help="Path to the trained checkpoint.",
    )
    parser.add_argument("--fallback", action="store_true",
                        help="Use ViT-Base instead of I-JEPA ViT-H.")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # --- Load checkpoint ----------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Rebuild model
    projector = ProjectorMLP()
    projector.load_state_dict(ckpt["projector_state"])

    caption_model = CaptionModel(projector, freeze_gpt2=True)

    # Optionally load fine-tuned GPT-2 weights
    if ckpt.get("gpt2_state") is not None:
        caption_model.gpt2.load_state_dict(ckpt["gpt2_state"])
        print("   Loaded fine-tuned GPT-2 weights.")

    caption_model = caption_model.to(device).eval()

    # --- Load encoder and extract feature -----------------------------------
    encoder, processor, actual_dim = load_encoder(use_fallback=args.fallback)
    encoder = encoder.to(device).eval()

    print(f"\nProcessing: {args.image}")
    feature = extract_single_feature(args.image, encoder, processor, actual_dim, device)

    # --- Generate caption ---------------------------------------------------
    caption = caption_model.generate(
        feature,
        max_new_tokens=args.max_tokens,
        num_beams=args.num_beams,
    )
    print(f"\n📝  Caption:  {caption}")


if __name__ == "__main__":
    main()
