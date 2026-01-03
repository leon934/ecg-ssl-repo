from pathlib import Path

from torchvision.transforms import transforms
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import av
from tqdm import tqdm

np.random.seed(0)

# creates 2 separate views per simclr paper
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.n_views = 2

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]
    
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

# dataset wrapper to pass into pytorch dataloader class
class FrameDataset(Dataset):
    def __init__(self, array, length, transform):
        self.array = array
        self.length = length
        self.transform = transform

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.transform(self.array[index])

class ContrastiveLearningDataset:
    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomResizedCrop(size=size),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms
    
    def get_dataset(self, name):
        # build dataset based on path (and name for scalability)
        # must lead to the root /EchoNet-Dynamic folder
        if name == "echonet-dynamic":
            # derived from ipynb, so the arrays can be created beforehand in mem. w/ np vs using python lists
            # only issue is scalability if the dataset is updated for wtv reason
            NUM_TRAIN_FRAMES = 1315340
            NUM_VAL_FRAMES = 228836
            NUM_TEST_FRAMES = 226460

            VID_DIM = 112

            if (dataset_path := self.root_folder / "echonet-np_arr.npz").exists():
                print("Found saved EchoNet-Dynamic dataset. Skipping creating views.")
                dataset_case = np.load(dataset_path, mmap_mode="r")
                
                return {
                    split_type: FrameDataset(
                        array=dataset_case[split_type],
                        length=dataset_case[f"{split_type}_IDX"],
                        transform=ContrastiveLearningViewGenerator(
                            self.get_simclr_pipeline_transform(VID_DIM)
                        )
                    ) for split_type in ("TRAIN", "VAL", "TEST")
                }

            video_name_df = pd.read_csv(self.root_folder / "FileList.csv")

            dataset_case = {
                "TRAIN": [np.zeros((NUM_TRAIN_FRAMES, VID_DIM, VID_DIM), dtype=np.uint8), 0],
                "VAL": [np.zeros((NUM_VAL_FRAMES,     VID_DIM, VID_DIM), dtype=np.uint8), 0],
                "TEST": [np.zeros((NUM_TEST_FRAMES,   VID_DIM, VID_DIM), dtype=np.uint8), 0]
            }
            
            # faster than loc
            split_map = dict(zip(video_name_df["FileName"], video_name_df["Split"]))

            # iterate through each .avi file
            for file_name in tqdm(video_name_df["FileName"]):
                obj = av.open(self.root_folder / "videos" / (file_name + ".avi"))

                curr_split = split_map[file_name]
                curr_arr, curr_idx = dataset_case[curr_split]

                # iterate through each frame and add it to the np arr
                for frame in obj.decode(video=0):
                    frame_arr = frame.to_ndarray(format="gray")

                    curr_arr[curr_idx] = frame_arr
                    curr_idx += 1
                
                # reupdate stored idx
                dataset_case[curr_split][1] = curr_idx
                obj.close()

            np.savez(self.root_folder / "echonet-np_arr.npz", **{k: v[0] for k, v in dataset_case.items()}, **{f"{k}_IDX": np.array(v[1]) for k, v in dataset_case.items()})
            print("Finished saving np array.")

            return {
                split_type: FrameDataset(
                    *dataset_case[split_type],
                    transform=ContrastiveLearningViewGenerator(
                        self.get_simclr_pipeline_transform(VID_DIM)
                    )
                ) for split_type in ("TRAIN", "VAL", "TEST")
            }