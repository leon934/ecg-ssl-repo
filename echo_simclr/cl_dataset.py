from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import random

import torch
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np

from data_processing.echonet_dynamic import echonet_dataset

np.random.seed(0)

# creates 2 separate views per simclr paper
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.n_views = 2

    def __call__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return [self.base_transform(x) for _ in range(self.n_views)]

# dataset wrapper to pass into pytorch dataloader class
class FrameDataset(Dataset):
    def __init__(self, array: torch.Tensor, transform: Callable, **kwargs):
        if kwargs:
            raise TypeError(
                f"FrameDataset got unexpected arguments: {kwargs.keys()}"
            )

        self.array = array
        self.length = len(self.array)
        self.transform = transform

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index) -> torch.Tensor:
        curr_video = self.array[index]
        sampled_frame_idx = random.randint(0, len(curr_video) - 1)

        return self.transform(curr_video[sampled_frame_idx]), 0

class VideoDataset(Dataset):
    def __init__(self, array: torch.Tensor, transform: Callable, clip_length: int):
        # [T, C, H, W]
        self.vid_array = array
        self.length = len(self.vid_array)
        self.transform = transform
        self.clip_length = clip_length

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index) -> torch.Tensor:
        # [T, C, H, W]
        curr_video = self.vid_array[index]
        
        # assumes all videos >= clip_len due to dataset proc in echonet_dynamic.py
        end_idx = len(curr_video) - self.clip_length
        clip_start_idx = random.randint(0, end_idx)

        curr_clip = curr_video[clip_start_idx : clip_start_idx + self.clip_length]

        # # todo: maybe faster way to iterate thru frames such as vectorizing?
        for i, frame in enumerate(curr_clip):
            curr_clip[i] = self.transform(frame)

        return curr_clip

class ContrastiveLearningDataset:
    @dataclass(frozen=True)
    class DatasetSpec:
        transform_func: Callable
        dataset_class: object

    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)

    @staticmethod
    def get_simclr_frame_pipeline_transform(size, s=1) -> transforms.Compose:
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1)]),
                                              transforms.ToTensor()])
        return data_transforms
    
    @staticmethod
    def get_simclr_video_pipeline_transform(size, s=1) -> transforms.Compose:
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transformations = []

        # fixes transformation for a given video
        if random.random() < 0.8:
            transformations.append(color_jitter)
        
        if random.random() < 0.5:
            transformations.append(transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1))

        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size)]
                                           + transformations
                                           + [transforms.ToTensor()])
        return data_transforms
    
    def get_dataset(self, name: str, args: dict)-> Union[FrameDataset, VideoDataset]:
        # build dataset based on path (and name for scalability)
        # must lead to the root /EchoNet-Dynamic folder
        model_spec = {
            "vit": self.DatasetSpec(
                transform_func=self.get_simclr_frame_pipeline_transform,
                dataset_class=FrameDataset
            ),
            "vivit": self.DatasetSpec(
                transform_func=self.get_simclr_video_pipeline_transform,
                dataset_class=VideoDataset
            ) 
        }

        transform_func = model_spec[args.model].transform_func
        DatasetClass = model_spec[args.model].dataset_class
        
        valid_datasets = {
            "echonet-dynamic": lambda: DatasetClass(array=echonet_dataset(self.root_folder, args),
                                                    transform=ContrastiveLearningViewGenerator(
                                                        transform_func(112)
                                                    ),
                                                    clip_length=args.clip_length if args.model == "vivit" else None)
        }
        
        datasets_fn = valid_datasets[name]
        return datasets_fn()