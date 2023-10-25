import glob
import os
from abc import ABC
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from kornia.augmentation import ColorJiggle, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, \
    Resize
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from utils.utils import get_size_without_zero_padding


def get_transforms() -> Compose:
    return Compose([
        ToTensor()
    ])


def get_augmentation(crop_size: tuple):
    return Compose([
        ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25), hue=(-0.02, 0.02),
                    same_on_batch=True),
        RandomCrop(size=crop_size, same_on_batch=True),
        RandomVerticalFlip(same_on_batch=True),
        RandomHorizontalFlip(same_on_batch=True),
        RandomRotation(degrees=180, same_on_batch=True),
    ])


def augment_video(video: Tensor):
    crop_size = np.random.randint(video.shape[-2] // 3, video.shape[-2]), \
                np.random.randint(video.shape[-1] // 3, video.shape[-1])
    video = get_augmentation(crop_size)(video)
    x1, y1, x2, y2 = get_size_without_zero_padding(video[0])
    return video[:, :, y1:y2, x1:x2]


class VSRDataset(Dataset, ABC):
    def __init__(self, dataset_root: str, upscale_factor: float = 2.,
                 device: torch.device = torch.device("cuda")) -> None:
        """
        Dataset used for learning video superresolution. Dataloader with this dataset would return batches with
        shape B, N, C, H, W, where B is batch size, N number of frames of the longest video in batch
        :param dataset_root: path to dataset root [str]
        :param upscale_factor: up-scaling factor desired in superresolution algorithm
        """
        super().__init__()
        self.root = dataset_root
        self.device = device
        self.upscale_factor = upscale_factor
        self.transform = get_transforms()
        txt_files = list(filter(lambda filename: filename.endswith(".txt"), os.listdir(dataset_root)))
        self.video_txt_files = [os.path.join(self.root, filename) for filename in txt_files]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with open(self.video_txt_files[index]) as f:
            hr_video = []
            for path_to_frame in f.readlines():
                path_to_frame = path_to_frame.replace("\n", "")
                hr_frame = Image.open(os.path.join(self.root, path_to_frame)).convert("RGB")
                hr_video.append(self.transform(hr_frame))
        hr_video = torch.stack(hr_video).to(self.device)

        hr_video = augment_video(hr_video)
        lr_video = Resize((int(hr_video.shape[-2] / self.upscale_factor),
                           int(hr_video.shape[-1] / self.upscale_factor)))(hr_video)

        return hr_video, lr_video

    def __len__(self):
        return len(self.video_txt_files)


class SRDataset(Dataset, ABC):
    def __init__(self, dataset_root: str, sets: list[str], downscale_factor: float = 2):
        self.downscale_factor = downscale_factor
        self.transform = get_transforms()
        self.root = dataset_root
        self.image_paths = []
        self.filenames = []
        for subset in sets:
            set_path = os.path.join(dataset_root, subset, "image_SRF_2")
            for path in glob.glob(set_path + "/*HR.png"):
                self.image_paths.append(path)
                self.filenames.append(f"{subset}_{os.path.basename(path)[:-7]}")

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        frame_sr = Image.open(self.image_paths[index]).convert("RGB")
        frame = frame_sr.resize((int(frame_sr.width // self.downscale_factor),
                                 int(frame_sr.height // self.downscale_factor)))
        return self.transform(frame), self.transform(frame_sr)

    def __len__(self) -> int:
        return len(self.image_paths)
