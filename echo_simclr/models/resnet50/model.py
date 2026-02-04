import torch
import torch.nn as nn

from transformers import ResNetConfig, ResNetModel as RNModel

class ResNetModel(nn.Module):
    def __init__(self, image_size: int, channels: int, finetune_mode: bool, **kwargs):
        super(ResNetModel, self).__init__()

        config = ResNetConfig(
            num_channels=channels,
        )

        self.backbone = RNModel(config)

        dim_mlp = config.hidden_sizes[-1]
        OUTPUT_DIM = 256

        if finetune_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

            self.head = nn.Linear(dim_mlp, 3)
        else:
            self.head = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(), 
                nn.Linear(dim_mlp, OUTPUT_DIM)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.flatten(self.backbone(x).pooler_output, 1)
        z = self.head(h)

        return z