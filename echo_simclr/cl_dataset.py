import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

from data_processing.echonet_dynamic import echonet_dataset

np.random.seed(0)

# creates 2 separate views per simclr paper
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.n_views = 2

    def __call__(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return [self.base_transform(x) for _ in range(self.n_views)]

# dataset wrapper to pass into pytorch dataloader class
class FrameDataset(Dataset):
    def __init__(
            self, 
            array: np.ndarray,
            target_var_array: Optional[np.ndarray],
            transform: Optional[Callable],
            **kwargs
        ):
        self.array = array
        # converts to correct dtype to prevent further issues down the line
        self.target_array = torch.tensor(target_var_array, dtype=torch.float32) if target_var_array is not None else None
        
        self.length = len(self.array)
        self.transform = transform

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        curr_video = self.array[index]
        sampled_frame_idx = random.randint(0, len(curr_video) - 1)

        frame = torch.from_numpy(curr_video[sampled_frame_idx])

        # unnecessary computation when pretraining, so we js store target_arr as None
        X_val = self.transform(frame) if self.transform is not None else frame
        Y_val = 0 if self.target_array is None else self.target_array[index]
        return X_val, Y_val

class VideoDataset(Dataset):
    def __init__(
            self,
            array: np.ndarray,
            target_var_array: Optional[np.ndarray],
            transform: Callable,
            clip_length: int,
        ):
        # [T, C, H, W]
        self.vid_array = array
        self.target_array = torch.tensor(target_var_array, dtype=torch.float32) if target_var_array is not None else None

        self.length = len(self.vid_array)
        self.transform = transform
        self.clip_length = clip_length

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        curr_video = self.vid_array[index]
        
        # assumes all videos >= clip_len due to dataset proc in echonet_dynamic.py
        end_idx = len(curr_video) - self.clip_length
        clip_start_idx = random.randint(0, end_idx)

        curr_clip = torch.from_numpy(curr_video[clip_start_idx : clip_start_idx + self.clip_length])
        curr_clip = tv_tensors.Video(curr_clip)

        # unnecessary computation when pretraining, so we js store target_arr as None]
        X_val = self.transform(curr_clip) if self.transform is not None else curr_clip
        Y_val = 0 if self.target_array is None else self.target_array[index]
        
        return X_val, Y_val

class ContrastiveLearningDataset:
    @dataclass
    class DatasetSplit:
        train: np.array=None
        val: np.array=None
        test: np.array=None

    _DatasetClass_dict = {
        "vit": FrameDataset,
        "vivit": VideoDataset,
        "resnet50": FrameDataset
    }
    # to add to available datasets, we follow arg convention of (root_folder, args)
    _dataset_dict = {
        "echonet-dynamic": lambda root_folder, args: echonet_dataset(root_folder, args)
    }

    def __init__(self, root_folder: str, model_type: str, dataset_name: str, addl_args: argparse.Namespace):
        self.args = addl_args

        self.root_folder = Path(root_folder)
        self.DatasetClass = self._DatasetClass_dict[model_type]

        # has ["FRAME_TRAIN", "EF_VAL", etc. for echonet-dataset keys]
        # essentially "sorts" the "FRAME_TRAIN", etc. into own DatasetSplit dataclass
        self.dataset_split_dict = defaultdict(self.DatasetSplit)

        if dataset_name == "echonet-dynamic":
            for type_split, arr in self._dataset_dict[dataset_name](self.root_folder, self.args).items():
                data, split = type_split.split("_")
                setattr(self.dataset_split_dict[data], split, arr)
                
    @staticmethod
    def _get_simclr_pipeline_transform(size, s=1) -> v2.Compose:
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size=size),
            v2.RandomApply([color_jitter], p=0.8),
            v2.RandomApply([v2.GaussianBlur(kernel_size=int(0.1 * size) | 1)], p=0.5)
        ])
        return data_transforms
    
    def get_dataset_split(
            self, 
            dataset_name: str, 
            split: str, 
            Y_name: Optional[str]=None
        )-> Union[FrameDataset, VideoDataset]:
        # build dataset based on path (and name for scalability)
        # must lead to the root /EchoNet-Dynamic folder
        valid_datasets = {
            "echonet-dynamic": lambda: self.DatasetClass(
                array=getattr(self.dataset_split_dict["X"], split),
                target_var_array=getattr(self.dataset_split_dict[Y_name], split) if Y_name is not None else None,
                transform=ContrastiveLearningViewGenerator(self._get_simclr_pipeline_transform(size=112)) if Y_name is None else None,
                clip_length=self.args.clip_length if self.args.model == "vivit" else None)
        }
        
        datasets_fn = valid_datasets[dataset_name]
        return datasets_fn()