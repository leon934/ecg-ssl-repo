import torch
import torch.nn as nn

from torchvision.models.vision_transformer import vit_b_16, vit_l_16, vit_h_14

model_dict = {
    "vit_b_16": lambda **kwargs : vit_b_16(**kwargs),
    "vit_l_16": lambda **kwargs : vit_l_16(**kwargs),
    "vit_h_14": lambda **kwargs : vit_h_14(**kwargs),
}

class ViTModel(nn.Module):
    def __init__(self, image_size: int, channels: int, **kwargs):
        super(ViTModel, self).__init__()

        base_model = kwargs["base_model"]

        model_fn = model_dict[base_model]
        img_dim, num_channels = image_size, channels

        self.backbone = model_fn(image_size=img_dim)

        old_conv = self.backbone.conv_proj
        self.backbone.conv_proj = nn.Conv2d(
            in_channels=num_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        OUTPUT_DIM = 256

        # add mlp head to vit
        dim_mlp = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, OUTPUT_DIM))

    def forward(self, x) -> torch.Tensor:
        return self.backbone(x)