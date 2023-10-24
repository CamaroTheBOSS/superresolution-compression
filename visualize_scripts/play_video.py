import os

import cv2
import numpy as np
import torch
from PIL import Image
from kornia.augmentation import ColorJiggle, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomCrop, \
    Resize
from torch import Tensor
from torchvision.transforms import Compose

from models.VSRDataset import get_transforms


def augmentation(video: Tensor):
    def get_augmentation():
        crop_size = np.random.randint(video.shape[-2] // 3, video.shape[-2]), \
                    np.random.randint(video.shape[-1] // 3, video.shape[-1])
        print(crop_size)
        return Compose([
            ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25), hue=(-0.02, 0.02),
                        same_on_batch=True),
            RandomCrop(size=crop_size, same_on_batch=True),
            RandomVerticalFlip(same_on_batch=True),
            RandomHorizontalFlip(same_on_batch=True),
            RandomRotation(degrees=180, same_on_batch=True),
        ])

    # RandomCrop(size=(video.shape[-2], video.shape[-1]), same_on_batch=True)
    augment = get_augmentation()
    transformed_video = augment(video)
    transformed_video = Resize(size=(video.shape[-2], video.shape[-1]))(transformed_video)

    return video, transformed_video


@torch.no_grad()
def play_video(path_to_txt_file: str, video_transform: callable = lambda x: (x, x), fps: int = 30, repeat=True):
    with open(path_to_txt_file) as f:
        root = os.path.dirname(path_to_txt_file)
        transforms = get_transforms()
        video = []
        for path_to_frame in f.readlines():
            path_to_frame = path_to_frame.replace("\n", "")
            frame = Image.open(os.path.join(root, path_to_frame)).convert("RGB")
            video.append(transforms(frame))
    video = torch.stack(video)
    video, transformed_video = video_transform(video)
    new_transform = False
    while repeat:
        if new_transform:
            video, transformed_video = video_transform(video)
            new_transform = False
        for frame, transformed_frame in zip(video, transformed_video):
            output = torch.concatenate((frame, transformed_frame), dim=2)
            output = output.detach().cpu().permute(1, 2, 0).numpy()
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imshow("window", output)
            if cv2.waitKey(1000 // fps) & 0xff == ord('q'):
                print("New transform will be prepared at the next start of the video")
                new_transform = True


if __name__ == "__main__":
    play_video("./datasets/VSR/VID4/walk.txt", video_transform=augmentation)
