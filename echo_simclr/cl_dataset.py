from pathlib import Path

from torchvision.transforms import transforms
from torchvision import transforms, datasets
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

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]

# dataset wrapper to pass into pytorch dataloader class
class FrameDataset(Dataset):
    def __init__(self, array, length, transform):
        self.array = array
        self.length = length
        self.transform = transform

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.transform(self.array[index]), 0

class ContrastiveLearningDataset:
    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
                                              transforms.RandomResizedCrop(size=size),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1)]),
                                              transforms.ToTensor()])
        return data_transforms
    
    def get_dataset(self, name: str):
        # build dataset based on path (and name for scalability)
        # must lead to the root /EchoNet-Dynamic folder
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(self.root_folder,
                                                        train=True,
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(32)
                                                        ),
                                                        download=True),
            "echonet-dynamic": lambda: FrameDataset(array=echonet_dataset(self.root_folder),
                                                    length=echonet_dataset(self.root_folder).shape[0],
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(112)
                                                    ))
        }
        
        datasets_fn = valid_datasets[name]

        return datasets_fn()