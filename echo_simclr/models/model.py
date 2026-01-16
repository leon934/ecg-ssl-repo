from dataclasses import dataclass
from typing import Union

from models.vit.model import ViTModel
from models.vivit.model import ViViTModel

@dataclass(frozen=True)
class DatasetSpec:
    image_size: int
    channels: int

def get_model(model_name: str, dataset_name: str, clip_length: int, arch_type: str=None) -> Union[ViTModel, ViViTModel]:
    valid_datasets = {
        "cifar10": DatasetSpec(image_size=32, channels=3),
        "echonet-dynamic": DatasetSpec(image_size=112, channels=1)
    }
    
    model_dict = {
        "vit": ViTModel,
        "vivit": ViViTModel
    }

    model = model_dict[model_name](
        image_size=valid_datasets[dataset_name].image_size,
        channels=valid_datasets[dataset_name].channels,
        base_model=arch_type,
        clip_length=clip_length
    )

    return model