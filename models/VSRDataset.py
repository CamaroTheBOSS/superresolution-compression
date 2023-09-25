import os
from abc import ABC
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def get_transforms() -> Compose:
    return Compose([
        ToTensor(),
    ])


def collate_fn(inputs) -> dict:
    batch = {"LR": torch.stack([i[0] for i in inputs], dim=0), "HR": torch.stack([i[1] for i in inputs], dim=0)}
    return batch


class Video:
    def __init__(self, path_to_txt: str) -> None:
        self.path_to_txt = path_to_txt
        self.frames = []
        with open(path_to_txt, "r") as file:
            for relative_path in file.readlines():
                self.frames.append(os.path.join(Path(path_to_txt).parent, relative_path[:-1]))

    def __len__(self):
        return len(self.frames)


class VSRDataset(Dataset, ABC):
    def __init__(self, dataset_root: str, downscale_factor: float = 2, test=False) -> None:
        super().__init__()
        self.root = dataset_root
        self.downscale_factor = downscale_factor
        self.transform = get_transforms()
        txt_files = list(filter(lambda filename: filename.endswith(".txt"), os.listdir(dataset_root)))
        self.videos = [Video(os.path.join(dataset_root, file)) for file in txt_files]
        self.videos = [self.videos[-1]] if test else self.videos[:-1]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        frame_sr = Image.open(self.get_next_frame_path(index)).convert("RGB")
        frame = frame_sr.resize((int(frame_sr.width // self.downscale_factor),
                                 int(frame_sr.height // self.downscale_factor)))
        frame = self.transform(frame)
        frame_sr = self.transform(frame_sr)

        return frame, frame_sr

    def __len__(self) -> int:
        return sum([len(video) for video in self.videos])

    def get_next_frame_path(self, frame_idx: int) -> str:
        cumulative_length = 0
        video_idx = 0
        for i in range(0, len(self.videos)):
            if frame_idx < cumulative_length + len(self.videos[i]):
                video_idx = i
                break
            cumulative_length += len(self.videos[i])

        path = self.videos[video_idx].frames[frame_idx - cumulative_length]
        # print(f"Getting frame {frame_idx} from video {video_idx}")
        return path
