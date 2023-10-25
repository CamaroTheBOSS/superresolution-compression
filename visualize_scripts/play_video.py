import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from kornia.augmentation import ColorJiggle, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomCrop, \
    Resize
from torch import Tensor
from torchvision.transforms import Compose

from models.VSRDataset import get_transforms, augment_video


def augmentation(video: Tensor):
    transformed_video = augment_video(video.to(torch.device("cuda")))
    transformed_video = Resize(size=(video.shape[-2], video.shape[-1]))(transformed_video)

    return video, transformed_video.to(torch.device("cpu"))


@torch.no_grad()
def play_video(path_to_txt_file: str, video_transform: callable = lambda x: (x, x), fps: int = 30, repeat=True):
    """
    Function, which transform video from specified path with given callback function and plays it in cv2 window.

    Note that callback function must take video Tensor with shape N, C, H, W where N is number of frames in the video
    and should return tuple with original video Tensor at the first place and transformed video Tensor at the second
    place. Both Tensors should have the same shape N, C, H, W to ensure that this function will visualize well.

    It's possible to restart the visualization. It is needed to click 'Q' at the keyboard. If so, transform will be
    repeated at the next start of the video visualization.

    :param path_to_txt_file: string path to text file with paths to video frame images
    :param video_transform: function which takes video Tensor with shape N, C, H, W where N is number of frames in the
    video and returns tuple with original video Tensor at the first place and transformed video Tensor at the second
    place. Both Tensors should have the same shape N, C, H, W to ensure that this function will visualize well.
    :param fps: Speed of visualization
    :param repeat: If true, visualization will be repeated after finishing
    :return:
    """
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
