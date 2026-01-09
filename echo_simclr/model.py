import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, vit_l_16, vit_h_14

model_dict = {
    "vit_b_16": lambda **kwargs : vit_b_16(**kwargs),
    "vit_l_16": lambda **kwargs : vit_l_16(**kwargs),
    "vit_h_14": lambda **kwargs : vit_h_14(**kwargs),
}

# dataset: (image dim, # channels)
valid_datasets = {
    "cifar10": (32, 3),
    "echonet-dynamic": (112, 1)
}

class ViTModel(nn.Module):
    def __init__(self, base_model, dataset_name):
        super(ViTModel, self).__init__()

        model_fn = model_dict[base_model]
        img_dim, num_channels = valid_datasets[dataset_name]

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

        # add mlp head to vit
        dim_mlp = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 256))

    def forward(self, x):
        return self.backbone(x)