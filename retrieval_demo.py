#!/usr/bin/env python3
"""
retrieval_demo.py — Visual similarity retrieval using I-JEPA embeddings.

Demonstrates that I-JEPA's self-supervised feature vectors capture
meaningful visual/semantic similarity between images — with zero
text or language involved.  This is a standalone side-demo, independent
of the captioning pipeline.

All it needs is the pre-computed features.pt cache (produced by
extract_features.py) and the raw images on disk.

Usage:
    python retrieval_demo.py --query 1000268201_693b08cb0e.jpg
    python retrieval_demo.py --top_k 8          # random query, 8 neighbours
    python retrieval_demo.py                    # random query, 5 neighbours
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

# matplotlib is imported lazily inside build_grid() so the script can
# still print results even if matplotlib is missing.


# ---------------------------------------------------------------------------
# Core retrieval logic
# ---------------------------------------------------------------------------
def load_features(path: str):
    """
    Load the cached feature dict and return parallel lists of
    (filenames, feature_matrix).

    Returns:
        filenames: list[str]           — N image filenames
        matrix:    Tensor [N, 1280]    — float32 feature matrix
    """
    features = torch.load(path, map_location="cpu", weights_only=True)
    filenames = sorted(features.keys())
    matrix = torch.stack([features[f].float() for f in filenames], dim=0)
    return filenames, matrix


def retrieve_neighbours(
    query_idx: int,
    feature_matrix: torch.Tensor,
    top_k: int = 5,
):
    """
    Find the top-K most similar images to the query using cosine similarity.

    All similarity scores are computed in a single vectorized matrix
    operation — no Python loops over the ~8000 images.

    Args:
        query_idx:      index of the query image in feature_matrix
        feature_matrix: [N, D] float32 tensor
        top_k:          number of neighbours to return

    Returns:
        indices:      list[int]   — indices of the top-K neighbours
        similarities: list[float] — corresponding cosine similarity scores
    """
    # Query vector: [1, D]
    query_vec = feature_matrix[query_idx].unsqueeze(0)

    # Cosine similarity against every image: [N]
    sims = F.cosine_similarity(query_vec, feature_matrix, dim=1)

    # Exclude the query itself by setting its similarity to -inf
    sims[query_idx] = -float("inf")

    # Top-K
    top_sims, top_idxs = torch.topk(sims, k=min(top_k, len(sims)))

    return top_idxs.tolist(), top_sims.tolist()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def build_grid(
    query_filename: str,
    neighbour_filenames: list,
    neighbour_scores: list,
    images_dir: str,
    output_path: str,
):
    """
    Build and save a matplotlib figure:
        [QUERY] | [#1  sim=0.93] | [#2  sim=0.91] | …

    The query image is on the far left with a coloured border; neighbours
    follow in descending similarity order, each labelled with its score.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n_images = 1 + len(neighbour_filenames)
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4.5))

    if n_images == 1:
        axes = [axes]

    # --- Query image -------------------------------------------------------
    query_path = os.path.join(images_dir, query_filename)
    try:
        query_img = Image.open(query_path).convert("RGB")
    except Exception as e:
        print(f"   ⚠️  Could not open query image {query_path}: {e}")
        query_img = Image.new("RGB", (224, 224), color=(40, 40, 40))

    axes[0].imshow(query_img)
    axes[0].set_title("QUERY", fontsize=13, fontweight="bold", color="#2196F3")
    axes[0].set_xlabel(query_filename, fontsize=7, color="gray")
    axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # Blue border for the query
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#2196F3")
        spine.set_linewidth(3)

    # --- Neighbour images --------------------------------------------------
    for i, (fname, score) in enumerate(zip(neighbour_filenames, neighbour_scores)):
        ax = axes[i + 1]
        img_path = os.path.join(images_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"   ⚠️  Could not open {img_path}: {e}")
            img = Image.new("RGB", (224, 224), color=(40, 40, 40))

        ax.imshow(img)
        ax.set_title(f"#{i + 1}  sim = {score:.4f}", fontsize=11)
        ax.set_xlabel(fname, fontsize=7, color="gray")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#ccc")
            spine.set_linewidth(1)

    fig.suptitle(
        "I-JEPA Embedding Retrieval — Nearest Neighbours (cosine similarity)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📊  Saved grid to {output_path}")

    # Show interactively if possible (won't block in non-interactive envs)
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve visually similar images using I-JEPA embeddings. "
            "No text or language model involved — pure visual similarity."
        ),
    )
    parser.add_argument(
        "--features",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "features.pt"),
        help="Path to the cached features.pt file.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "Images"),
        help="Path to the directory of .jpg images.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Filename of the query image (must be a key in features.pt). "
             "If omitted, a random image is chosen.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of nearest neighbours to retrieve.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "retrieval_result.png"),
        help="Where to save the result grid image.",
    )
    args = parser.parse_args()

    # --- Load features ------------------------------------------------------
    print(f"Loading features from {args.features} …")
    filenames, matrix = load_features(args.features)
    print(f"   {len(filenames)} images, feature dim = {matrix.shape[1]}")

    # --- Select query -------------------------------------------------------
    if args.query is not None:
        if args.query not in filenames:
            print(f"❌  '{args.query}' not found in features.pt.")
            print(f"   Available keys look like: {filenames[:3]}")
            return
        query_idx = filenames.index(args.query)
    else:
        query_idx = random.randint(0, len(filenames) - 1)

    query_filename = filenames[query_idx]
    print(f"\n🔍  Query image: {query_filename}")

    # --- Retrieve neighbours ------------------------------------------------
    top_idxs, top_sims = retrieve_neighbours(query_idx, matrix, top_k=args.top_k)

    print(f"\nTop-{args.top_k} nearest neighbours (cosine similarity):\n")
    print(f"   {'Rank':<6} {'Filename':<40} {'Similarity':>10}")
    print(f"   {'─' * 6} {'─' * 40} {'─' * 10}")
    neighbour_filenames = []
    for rank, (idx, sim) in enumerate(zip(top_idxs, top_sims), start=1):
        fname = filenames[idx]
        neighbour_filenames.append(fname)
        print(f"   {rank:<6} {fname:<40} {sim:>10.4f}")

    # --- Build visual grid --------------------------------------------------
    print(f"\nBuilding visual grid …")
    build_grid(
        query_filename=query_filename,
        neighbour_filenames=neighbour_filenames,
        neighbour_scores=top_sims,
        images_dir=args.images_dir,
        output_path=args.output,
    )

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
