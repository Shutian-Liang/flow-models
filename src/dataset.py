import os
import torch
import math
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

class CIFAR10Dataset(LightningDataModule):
    """
    PyTorch Lightning DataModule for CIFAR-10 dataset.

    Args:
    data_dir: root directory of your dataset.
    train_batch_size: the batch size to use during training.
    val_batch_size: the batch size to use during validation.
    patch_size: the size of the crop to take from the original images.
    num_workers: the number of parallel workers to create to load data
        items (see PyTorch's Dataloader documentation for more details).
    pin_memory: whether prepared items should be loaded into pinned memory
        or not. This can improve performance on GPUs.
        
    """
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (32, 32),
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
        ):
        super().__init__()
        
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def setup(self):
        train_transforms = transforms.Compose([
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                            ])
        

        self.train_dataset = CIFAR10(root=self.data_dir, train=True, download=False, transform=train_transforms)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def show_images(self, images, epoch, path=None):
        images = images.detach().cpu().numpy()  # Convert images to numpy arrays
        fig = plt.figure(figsize=(4, 4))
        cols = math.ceil(len(images) ** (1 / 2))
        rows = math.ceil(len(images) / cols)

        for r in range(rows):
            for c in range(cols):
                idx = cols * r + c
                ax = fig.add_subplot(rows, cols, idx + 1)
                ax.axis('off')
                if idx < len(images):
                    # Transpose the image dimensions from (C, H, W) to (H, W, C)
                    img = images[idx].transpose((1, 2, 0))
                    # Display the image without applying a color map
                    ax.imshow(img)
        fig.suptitle(f'epoch {epoch}', fontsize=18)
        plt.show()
        plt.savefig(path) if path else None
    
