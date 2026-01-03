import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

vit_dict = {
    "vit_b_16": vit_b_16(),
    "vit_b_32": vit_b_32(),
    "vit_l_16": vit_l_16(),
    "vit_l_32": vit_l_32(),
    "vit_h_14": vit_h_14(),
}

class ViTModel(nn.Module):
    def __init__(self, base_model):
        super(ViTModel, self).__init__()

        self.backbone = vit_dict[base_model]

        # add mlp head to vit
        dim_mlp = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.heads.head)

    def forward(self, x):
        return self.backbone(x)