import torch
import torch.nn as nn

# naming convention to prevent dupe import
from transformers import ViTModel as VitModel, ViTConfig

class ViTModel(nn.Module):
    # "overloading" init with clip_length to make code generalizable to clip-based encoders (e.g. ViViT)
    def __init__(self, image_size: int, channels: int, eval_mode: bool=False, **kwargs):
        super(ViTModel, self).__init__()

        config = ViTConfig(
            image_size=image_size,
            num_channels=channels
        )

        self.backbone = VitModel(config)
        self.backbone.classifier = nn.Identity()

        # add mlp head to vit
        dim_mlp = config.hidden_size
        OUTPUT_DIM = 256
        
        if eval_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

            self.head = nn.Linear(dim_mlp, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                nn.Linear(dim_mlp, OUTPUT_DIM)
            )
            
    def forward(self, x) -> torch.Tensor:
        h = self.backbone(x).last_hidden_state[:, 0]
        z = self.head(h)

        return z