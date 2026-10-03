#!/usr/bin/env python3
"""
model.py — Projector MLP + CaptionModel for ClipCap-style prefix-tuning.

Architecture overview:
    image features (1280-d)  →  Projector MLP  →  8 pseudo-token embeddings (768-d each)
    pseudo-tokens ++ caption token embeddings  →  GPT-2 LM head  →  caption

The Projector is the only trainable component by default.
GPT-2 can optionally be unfrozen for fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ---------------------------------------------------------------------------
# Projector MLP
# ---------------------------------------------------------------------------
class ProjectorMLP(nn.Module):
    """
    Maps a single 1280-dim image feature vector into `n_prefix` token
    embeddings of size `gpt2_dim`.

    Architecture:
        Linear(1280 → hidden)  →  GELU  →  Linear(hidden → n_prefix * gpt2_dim)
        → reshape to [batch, n_prefix, gpt2_dim]

    Parameters (~200 K for default config):
        hidden=512, n_prefix=8, gpt2_dim=768
        1280*512 + 512 + 512*(8*768) + 8*768 ≈ 3.8M … let's use a smaller hidden
        With hidden=256: 1280*256 + 256 + 256*6144 + 6144 ≈ 1.9M
        Still a bit chunky. Keep it — it's a small fraction of GPT-2's 124M.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        gpt2_dim: int = 768,
        n_prefix: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_prefix = n_prefix
        self.gpt2_dim = gpt2_dim

        output_dim = n_prefix * gpt2_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_dim]
        returns: [batch, n_prefix, gpt2_dim]
        """
        out = self.net(x)  # [batch, n_prefix * gpt2_dim]
        return out.view(-1, self.n_prefix, self.gpt2_dim)


# ---------------------------------------------------------------------------
# Caption Model (wraps GPT-2 + Projector)
# ---------------------------------------------------------------------------
class CaptionModel(nn.Module):
    """
    ClipCap-style captioning model.

    Training forward pass:
        1. Project image features → prefix embeddings  [B, P, 768]
        2. Embed caption tokens via GPT-2's wte        [B, T, 768]
        3. Concatenate prefix + caption embeddings      [B, P+T, 768]
        4. Run through GPT-2 with labels masked:
           prefix positions → -100 (ignored by loss)
           caption positions → token ids

    Inference (generate):
        Build prefix embeddings, then call GPT-2's .generate() with
        inputs_embeds as the initial context.
    """

    def __init__(
        self,
        projector: ProjectorMLP,
        gpt2_model_name: str = "gpt2",
        freeze_gpt2: bool = True,
    ):
        super().__init__()
        self.projector = projector
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

        # GPT-2 doesn't have a pad token by default — use eos
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id

        if freeze_gpt2:
            self.freeze_gpt2()

    def freeze_gpt2(self):
        """Freeze all GPT-2 parameters."""
        for p in self.gpt2.parameters():
            p.requires_grad = False

    def unfreeze_gpt2(self):
        """Unfreeze GPT-2 parameters for fine-tuning."""
        for p in self.gpt2.parameters():
            p.requires_grad = True

    def forward(
        self,
        image_features: torch.Tensor,   # [B, 1280]
        input_ids: torch.Tensor,        # [B, T]  (caption token ids)
        attention_mask: torch.Tensor,    # [B, T]
    ) -> torch.Tensor:
        """
        Compute the language modelling loss over caption tokens only.

        Returns the scalar cross-entropy loss.
        """
        batch_size = image_features.shape[0]
        n_prefix = self.projector.n_prefix

        # 1. Project image features → prefix embeddings
        prefix_embeds = self.projector(image_features)  # [B, P, 768]

        # 2. Embed caption tokens
        caption_embeds = self.gpt2.transformer.wte(input_ids)  # [B, T, 768]

        # 3. Concatenate
        inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)  # [B, P+T, 768]

        # 4. Build attention mask (1 for all prefix positions + original mask)
        prefix_mask = torch.ones(
            batch_size, n_prefix,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 5. Build labels: -100 for prefix (ignored), token ids for caption
        ignore_prefix = torch.full(
            (batch_size, n_prefix),
            fill_value=-100,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        labels = torch.cat([ignore_prefix, input_ids], dim=1)
        # Also mask out padding in labels
        labels = labels.masked_fill(full_attention_mask == 0, -100)

        # 6. Forward through GPT-2
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        image_features: torch.Tensor,   # [1, 1280]  (single image)
        max_new_tokens: int = 35,
        num_beams: int = 4,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate a clean caption for a single image.
        """
        self.eval()

        # Build prefix embeddings as the initial context
        prefix_embeds = self.projector(image_features)  # [1, P, 768]

        # Generate using beam search with repetition penalty
        output_ids = self.gpt2.generate(
            inputs_embeds=prefix_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=1.2,
            temperature=temperature,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode (skip special tokens)
        raw_caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Clean sentence boundary: stop at the first period
        if "." in raw_caption:
            idx = raw_caption.find(".")
            # Ensure it's not a short prefix like "A." or "Dr."
            if idx > 8:
                raw_caption = raw_caption[:idx + 1]

        # Fix tokenization artifacts like "word ." -> "word."
        cleaned = raw_caption.replace(" .", ".").replace(" ,", ",").replace(" '", "'").strip()
        return cleaned


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building model …")
    proj = ProjectorMLP()
    model = CaptionModel(proj)

    n_proj = sum(p.numel() for p in proj.parameters())
    n_gpt2 = sum(p.numel() for p in model.gpt2.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Projector params:  {n_proj:,}")
    print(f"GPT-2 params:      {n_gpt2:,}")
    print(f"Trainable params:  {n_train:,}")

    # Dummy forward
    dummy_feat = torch.randn(2, 1280)
    dummy_ids = torch.randint(0, 50257, (2, 20))
    dummy_mask = torch.ones_like(dummy_ids)
    loss = model(dummy_feat, dummy_ids, dummy_mask)
    print(f"Dummy loss: {loss.item():.4f}")

    # Dummy generate
    caption = model.generate(torch.randn(1, 1280))
    print(f"Dummy caption: {caption}")
