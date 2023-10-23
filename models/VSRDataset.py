import glob
import os
from abc import ABC
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor


class NormalizeFromMinus1To1:
    def __init__(self):
        pass

    def __call__(self, input_tensor):
        return input_tensor * 2 - 1


def get_transforms() -> Compose:
    return Compose([
        ToTensor(),
        NormalizeFromMinus1To1()
    ])


def get_max_size_from_inputs(inputs):
    max_size_lr, max_size_hr = (0, 0), (0, 0)
    for low_res, high_res in inputs:
        lr_shape = low_res.shape[-2:]
        hr_shape = high_res.shape[-2:]
        max_size_lr = max(max_size_lr[0], lr_shape[0]), max(max_size_lr[1], lr_shape[1])
        max_size_hr = max(max_size_hr[0], hr_shape[0]), max(max_size_hr[1], hr_shape[1])
    return max_size_lr, max_size_hr


def align_images_with_pad(inputs):
    max_size_lr, max_size_hr = get_max_size_from_inputs(inputs)
    for e, (low_res, high_res) in enumerate(inputs):
        lr_shape = low_res.shape[-2:]
        hr_shape = high_res.shape[-2:]

        pad_updown, pad_side = (max_size_lr[0] - lr_shape[0]), (max_size_lr[1] - lr_shape[1])
        low_res_padded = pad(low_res, (0, pad_side, 0, pad_updown), value=-1)

        pad_updown, pad_side = (max_size_hr[0] - hr_shape[0]), (max_size_hr[1] - hr_shape[1])
        high_res_padded = pad(high_res, (0, pad_side, 0, pad_updown), value=-1)

        inputs[e] = (low_res_padded, high_res_padded)

    return inputs


def collate_fn(inputs) -> dict:
    lr_shapes = torch.stack([torch.Tensor(tuple(i[0].shape[-2:])) for i in inputs])
    hr_shapes = torch.stack([torch.Tensor(tuple(i[0].shape[-2:])) for i in inputs])
    aligned_inputs = align_images_with_pad(inputs)
    batch = {"LR": torch.stack([i[0] for i in aligned_inputs], dim=0),
             "HR": torch.stack([i[1] for i in aligned_inputs], dim=0),
             "LR_shapes": lr_shapes,
             "HR_shapes": hr_shapes}
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
