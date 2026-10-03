#!/usr/bin/env python3
"""
train.py — Train the Projector MLP (and optionally fine-tune GPT-2).

Loads pre-extracted image features from features.pt and captions from
captions.txt, then trains the projector against frozen GPT-2 using
teacher-forced cross-entropy on caption tokens.

Usage:
    python train.py                        # train projector only
    python train.py --unfreeze_gpt2        # also fine-tune GPT-2
    python train.py --epochs 12 --lr 5e-5  # custom hyperparams
"""

import argparse
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from dataset import load_captions, build_pairs, train_val_split
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
# Dataset that serves pre-extracted features + tokenized captions
# ---------------------------------------------------------------------------
class CaptionFeatureDataset(Dataset):
    """
    Each item is (feature_vector [1280], input_ids [T], attention_mask [T]).

    Features come from the pre-computed cache; captions are tokenized on
    the fly (fast enough with GPT-2's BPE tokenizer).
    """

    def __init__(
        self,
        pairs: list,
        features: dict,
        tokenizer: GPT2Tokenizer,
        max_length: int = 64,
    ):
        # Filter pairs to only those whose image has a cached feature
        self.items = []
        for img_path, caption in pairs:
            fname = os.path.basename(img_path)
            if fname in features:
                self.items.append((fname, caption))

        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, caption = self.items[idx]

        # Feature vector (convert float16 → float32 for training)
        feat = self.features[fname].float()

        # Tokenize caption
        enc = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)          # [T]
        attention_mask = enc["attention_mask"].squeeze(0)  # [T]

        return feat, input_ids, attention_mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = get_device()
    print(f"Device: {device}")

    # --- Load data ----------------------------------------------------------
    project_dir = os.path.dirname(__file__)
    captions_path = args.captions_path or os.path.join(project_dir, "data", "captions.txt")
    images_dir = args.images_dir or os.path.join(project_dir, "data", "Images")
    features_path = args.features_path or os.path.join(project_dir, "features.pt")
    checkpoint_dir = args.checkpoint_dir or os.path.join(project_dir, "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading cached features …")
    features = torch.load(features_path, map_location="cpu", weights_only=True)
    print(f"   {len(features)} image features loaded.")

    print("Loading captions …")
    captions_dict = load_captions(captions_path)
    pairs = build_pairs(captions_dict, images_dir)
    train_pairs, val_pairs = train_val_split(pairs, val_frac=0.1, seed=42)
    print(f"   Train: {len(train_pairs)} pairs  |  Val: {len(val_pairs)} pairs")

    # --- Build model --------------------------------------------------------
    projector = ProjectorMLP(
        input_dim=1280,
        gpt2_dim=768,
        n_prefix=8,
        hidden_dim=256,
    )
    model = CaptionModel(projector, gpt2_model_name="gpt2", freeze_gpt2=True)

    if args.unfreeze_gpt2:
        print("🔓  Unfreezing GPT-2 for fine-tuning.")
        model.unfreeze_gpt2()

    model = model.to(device)

    # --- Datasets & loaders -------------------------------------------------
    tokenizer = model.tokenizer

    train_ds = CaptionFeatureDataset(train_pairs, features, tokenizer, max_length=args.max_length)
    val_ds = CaptionFeatureDataset(val_pairs, features, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS is happier with 0 workers
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --- Optimizer ----------------------------------------------------------
    param_groups = [
        {"params": model.projector.parameters(), "lr": args.lr},
    ]
    if args.unfreeze_gpt2:
        param_groups.append(
            {"params": model.gpt2.parameters(), "lr": args.gpt2_lr},
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # --- Training -----------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # ---- Train ---------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for feats, input_ids, attn_mask in pbar:
            feats = feats.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            loss = model(feats, input_ids, attn_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # ---- Validate ------------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for feats, input_ids, attn_mask in val_loader:
                feats = feats.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                loss = model(feats, input_ids, attn_mask)
                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)

        print(
            f"Epoch {epoch:3d}  |  "
            f"train loss {avg_train_loss:.4f}  |  "
            f"val loss {avg_val_loss:.4f}"
        )

        # ---- Checkpoint (best val loss) ------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            ckpt_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "projector_state": model.projector.state_dict(),
                    "gpt2_state": model.gpt2.state_dict() if args.unfreeze_gpt2 else None,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"   💾  Saved best checkpoint → {ckpt_path}  (val loss {avg_val_loss:.4f})")

    print(f"\n✅  Training complete.  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the image captioning projector.")

    # Paths
    parser.add_argument("--captions_path", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--features_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64,
                        help="Max caption token length (including padding).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the projector.")
    parser.add_argument("--gpt2_lr", type=float, default=2e-5,
                        help="Learning rate for GPT-2 (only used with --unfreeze_gpt2).")
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # GPT-2 fine-tuning
    parser.add_argument("--unfreeze_gpt2", action="store_true",
                        help="Also fine-tune GPT-2 at a lower learning rate.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
