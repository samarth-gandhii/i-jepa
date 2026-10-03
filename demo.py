#!/usr/bin/env python3
"""
demo.py — Minimal Gradio interface for the image captioning system.

Upload an image → see the generated caption.

Usage:
    python demo.py
    python demo.py --fallback   # use ViT-Base encoder
"""

import argparse
import os

import torch
import gradio as gr
from PIL import Image

from model import ProjectorMLP, CaptionModel
from generate import load_encoder, extract_single_feature, FEATURE_DIM


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
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
DEVICE = None
CAPTION_MODEL = None
ENCODER = None
PROCESSOR = None
ACTUAL_DIM = None


def load_models(checkpoint_path: str, use_fallback: bool):
    """Load the caption model and vision encoder once."""
    global DEVICE, CAPTION_MODEL, ENCODER, PROCESSOR, ACTUAL_DIM

    DEVICE = get_device()
    print(f"Device: {DEVICE}")

    # Caption model
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    projector = ProjectorMLP()
    projector.load_state_dict(ckpt["projector_state"])

    CAPTION_MODEL = CaptionModel(projector, freeze_gpt2=True)
    if ckpt.get("gpt2_state") is not None:
        CAPTION_MODEL.gpt2.load_state_dict(ckpt["gpt2_state"])
        print("   Loaded fine-tuned GPT-2 weights.")
    CAPTION_MODEL = CAPTION_MODEL.to(DEVICE).eval()

    # Vision encoder
    ENCODER, PROCESSOR, ACTUAL_DIM = load_encoder(use_fallback=use_fallback)
    ENCODER = ENCODER.to(DEVICE).eval()

    print("✅  Models loaded and ready.")


# ---------------------------------------------------------------------------
# Inference function for Gradio
# ---------------------------------------------------------------------------
def caption_image(image: Image.Image) -> str:
    """
    Called by Gradio when the user uploads an image.
    Returns the generated caption string.
    """
    if image is None:
        return "Please upload an image."

    # Save to a temp file (extract_single_feature expects a path)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f, format="JPEG")
        tmp_path = f.name

    try:
        feature = extract_single_feature(
            tmp_path, ENCODER, PROCESSOR, ACTUAL_DIM, DEVICE
        )
        caption = CAPTION_MODEL.generate(feature)
        return caption
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="🖼️ Image Captioning Demo") as demo:
        gr.Markdown(
            """
            # 🖼️ Image Captioning Demo
            **Architecture:** I-JEPA ViT-H/14 (frozen) → Projector MLP → GPT-2 (frozen)

            Upload an image and the model will generate a caption using the
            ClipCap-style prefix-tuning approach.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload an image",
                    height=400,
                )
                btn = gr.Button("✨ Generate Caption", variant="primary")

            with gr.Column(scale=1):
                caption_output = gr.Textbox(
                    label="Generated Caption",
                    lines=3,
                    interactive=False,
                )

        btn.click(fn=caption_image, inputs=image_input, outputs=caption_output)
        image_input.change(fn=caption_image, inputs=image_input, outputs=caption_output)

        gr.Markdown(
            """
            ---
            *Projector-only training on Flickr8k. This is a demo — caption
            quality is limited by the small dataset and frozen language model.*
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gradio demo for image captioning.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt"),
    )
    parser.add_argument("--fallback", action="store_true",
                        help="Use ViT-Base instead of I-JEPA ViT-H.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    args = parser.parse_args()

    load_models(args.checkpoint, args.fallback)

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
