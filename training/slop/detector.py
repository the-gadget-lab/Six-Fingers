from pathlib import Path

import torch

SNAPSHOT = next(
    Path(__file__).parents[2].joinpath(
        "data/hf/hub/models--buildborderless--CommunityForensics-DeepfakeDet-ViT/snapshots"
    ).glob("*")
)


class HFDetector(torch.nn.Module):
    """Community Forensics ViT-S/384. Single logit; P(fake) = sigmoid(logit + bias)."""

    def __init__(self, path=SNAPSHOT, logit_bias: float = 0.0):
        super().__init__()
        from transformers import ViTForImageClassification

        self.net = ViTForImageClassification.from_pretrained(path)
        self.logit_bias = logit_bias

    def forward(self, x):
        return self.net(pixel_values=x).logits.squeeze(-1) + self.logit_bias

    @torch.no_grad()
    def predict_proba(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.sigmoid(self(x))

    @property
    def head(self):
        return self.net.classifier

    @classmethod
    def from_checkpoint(cls, ckpt_path, **kw):
        m = cls(**kw)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        m.load_state_dict(state)
        return m
