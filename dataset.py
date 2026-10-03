#!/usr/bin/env python3
"""
dataset.py — Parse Flickr8k captions and build train / val splits.

Expected layout (created by download_data.py):
    data/captions.txt   — CSV with header "image,caption"
    data/Images/*.jpg   — the images

Public API:
    load_captions(captions_path) -> dict[str, list[str]]
        Maps each image filename to its list of captions.

    build_pairs(captions_dict, images_dir) -> list[tuple[str, str]]
        Flattens into (image_path, caption) pairs, dropping any
        filenames whose .jpg doesn't actually exist on disk.

    train_val_split(pairs, val_frac, seed) -> (train, val)
        Splits by *image* (not by pair) so no image leaks between sets.
"""

import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def load_captions(captions_path: str) -> Dict[str, List[str]]:
    """
    Read captions.txt and return {filename: [caption, …]}.

    The file is a CSV with a header row:
        image,caption
        1000268201_693b08cb0e.jpg,A child in a pink dress …
    """
    captions: Dict[str, List[str]] = defaultdict(list)

    with open(captions_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        assert header[0].strip().lower() == "image", (
            f"Expected 'image' as first column header, got '{header[0]}'"
        )

        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0].strip()
            caption = row[1].strip()
            if filename and caption:
                captions[filename].append(caption)

    return dict(captions)


def build_pairs(
    captions_dict: Dict[str, List[str]],
    images_dir: str = "",
) -> List[Tuple[str, str]]:
    """
    Flatten captions_dict into (image_path_or_filename, caption) pairs.

    If images_dir exists on disk, filters to images present on disk.
    If images_dir is missing/empty (e.g. running on Colab with cached features.pt),
    uses filenames directly without requiring raw .jpg files.
    """
    pairs: List[Tuple[str, str]] = []
    check_disk = bool(images_dir and os.path.isdir(images_dir))
    missing = 0

    for filename, caps in captions_dict.items():
        if check_disk:
            img_path = os.path.join(images_dir, filename)
            if not os.path.isfile(img_path):
                missing += 1
                continue
        else:
            img_path = filename

        for cap in caps:
            pairs.append((img_path, cap))

    if missing > 0:
        print(f"⚠️  Skipped {missing} filenames whose .jpg was not found on disk.")

    return pairs


def train_val_split(
    pairs: List[Tuple[str, str]],
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split pairs into train and val sets *by image* to avoid leakage.

    All five captions for a given image end up in the same split.
    """
    # Collect unique image paths
    images = sorted(set(p[0] for p in pairs))
    rng = random.Random(seed)
    rng.shuffle(images)

    n_val = max(1, int(len(images) * val_frac))
    val_images = set(images[:n_val])

    train = [(p, c) for p, c in pairs if p not in val_images]
    val = [(p, c) for p, c in pairs if p in val_images]

    return train, val


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    captions_path = os.path.join(data_dir, "captions.txt")
    images_dir = os.path.join(data_dir, "Images")

    caps = load_captions(captions_path)
    print(f"Loaded captions for {len(caps)} images.")

    pairs = build_pairs(caps, images_dir)
    print(f"Total (image, caption) pairs: {len(pairs)}")

    train, val = train_val_split(pairs)
    print(f"Train pairs: {len(train)}  |  Val pairs: {len(val)}")

    # Show a sample
    if train:
        img, cap = train[0]
        print(f"\nSample — {os.path.basename(img)}: {cap}")
