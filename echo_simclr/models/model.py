from dataclasses import dataclass
from typing import Union

from models.vit.model import ViTModel
from models.vivit.model import ViViTModel
from models.resnet50.model import ResNetModel

@dataclass(frozen=True)
class DatasetSpec:
    image_size: int
    channels: int

# global to be imported, so we don't have to update everywhere
valid_datasets = {
    "cifar10": DatasetSpec(image_size=32, channels=3),
    "echonet-dynamic": DatasetSpec(image_size=112, channels=1)
}
    
model_dict = {
    "vit": ViTModel,
    "vivit": ViViTModel,
    "resnet50": ResNetModel
}

def get_model(model_name: str, dataset_name: str, clip_length: int, finetune_mode: bool) -> Union[ViTModel, ViViTModel]:
    model = model_dict[model_name](
        image_size=valid_datasets[dataset_name].image_size,
        channels=valid_datasets[dataset_name].channels,
        clip_length=clip_length,
        finetune_mode=finetune_mode
    )

    return model