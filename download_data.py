#!/usr/bin/env python3
"""
download_data.py — Download Flickr8k dataset via kagglehub.

Pulls the dataset from kagglehub("adityajn105/flickr8k") and symlinks
(or copies) it into data/Images/*.jpg and data/captions.txt so that the
rest of the pipeline has a stable, predictable layout.
"""

import os
import shutil
import kagglehub


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    print("⬇️  Downloading Flickr8k via kagglehub …")
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    print(f"   Raw download at: {path}")

    # --- Locate the images directory and captions file ----------------------
    # kagglehub may nest files in subdirectories; walk to find them.
    images_src = None
    captions_src = None

    for root, dirs, files in os.walk(path):
        # Look for the Images folder (may be named "Images" or "images")
        for d in dirs:
            if d.lower() == "images":
                candidate = os.path.join(root, d)
                # Verify it actually contains .jpg files
                if any(f.endswith(".jpg") for f in os.listdir(candidate)):
                    images_src = candidate
        # Look for the captions file
        for f in files:
            if f.lower() == "captions.txt":
                captions_src = os.path.join(root, f)

    if images_src is None:
        raise FileNotFoundError(
            f"Could not find an 'Images' directory with .jpg files under {path}. "
            "Check the kagglehub download layout."
        )
    if captions_src is None:
        raise FileNotFoundError(
            f"Could not find 'captions.txt' under {path}. "
            "Check the kagglehub download layout."
        )

    print(f"   Found images at:   {images_src}")
    print(f"   Found captions at: {captions_src}")

    # --- Set up data/ directory with symlinks --------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    images_dst = os.path.join(DATA_DIR, "Images")
    captions_dst = os.path.join(DATA_DIR, "captions.txt")

    # Images directory
    if os.path.exists(images_dst):
        print(f"   {images_dst} already exists — skipping.")
    else:
        os.symlink(os.path.abspath(images_src), images_dst)
        print(f"   Linked {images_dst} → {images_src}")

    # Captions file
    if os.path.exists(captions_dst):
        print(f"   {captions_dst} already exists — skipping.")
    else:
        os.symlink(os.path.abspath(captions_src), captions_dst)
        print(f"   Linked {captions_dst} → {captions_src}")

    # Quick sanity check
    n_images = len([f for f in os.listdir(images_dst) if f.endswith(".jpg")])
    print(f"\n✅  Data ready: {n_images} images, captions at {captions_dst}")


if __name__ == "__main__":
    main()
