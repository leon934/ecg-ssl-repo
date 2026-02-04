import torch.nn as nn
import torch

from transformers import VivitModel, VivitConfig

class ViViTModel(nn.Module):
    def __init__(self, image_size: int, channels: int, finetune_mode: bool, **kwargs):
        super(ViViTModel, self).__init__()

        clip_length = kwargs["clip_length"]

        config = VivitConfig(
            image_size=image_size,
            num_channels=channels,
            num_frames=clip_length,
            add_pooling_layer=False
        )

        self.backbone = VivitModel(config)

        if hasattr(self.backbone, 'pooler') and self.backbone.pooler is not None:
            del self.backbone.pooler
            self.backbone.pooler = None

        dim_mlp = self.backbone.config.hidden_size
        OUTPUT_DIM = 256

        if finetune_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

            self.head = nn.Linear(dim_mlp, 3)
        else:
            self.head = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, OUTPUT_DIM)
            )

    def forward(self, x) -> torch.Tensor:
        h = self.backbone(x).last_hidden_state[:, 0]
        z = self.head(h)

        return z