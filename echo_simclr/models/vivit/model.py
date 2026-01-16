import torch.nn as nn
import torch

from transformers import VivitModel, VivitConfig

class ViViTModel(nn.Module):
    def __init__(self, image_size: int, channels: int, **kwargs):
        super(ViViTModel, self).__init__()

        clip_length = kwargs["clip_length"]

        config = VivitConfig(
            image_size=image_size,
            num_channels=channels,
            num_frames=clip_length,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            tubelet_size=[2, 16, 16],  # time, height, width
        )

        self.backbone = VivitModel(config)

        dim_mlp = self.backbone.config.hidden_size
        OUTPUT_DIM = 256

        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, OUTPUT_DIM)
        )

    def forward(self, x) -> torch.Tensor:
        out = self.backbone(x).last_hidden_state
        cls = out[:, 0]

        return self.projection_head(cls)