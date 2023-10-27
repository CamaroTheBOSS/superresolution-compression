from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as f
from torch import Tensor
from torch.nn import Module, Conv2d, Conv3d


@dataclass
class VSRModelOutput:
    reconstruction: Optional[Tensor] = None
    loss: Optional[dict] = None


class VSRModel(Module):
    def __init__(self, window_size: int = 4):
        super(VSRModel, self).__init__()
        self.window_size = window_size
        self.Conv3D_A = Conv3d(3, 64, kernel_size=(1, 9, 9), stride=1, padding=(0, 4, 4))
        self.Conv3D_B = Conv3d(64, 32, kernel_size=1, stride=1, padding=0)
        self.Conv3D_C = Conv3d(32, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))

    def forward(self, frames: Tensor, labels: Tensor = None) -> VSRModelOutput:
        B, N, C, H, W = frames.shape
        desired_size = (2 * H, 2 * W) if labels is None else (labels.shape[-2], labels.shape[-1])

        reconstructed_video = []
        for i in range(N // self.window_size + 1):
            low, high = i * self.window_size, min((i + 1) * self.window_size, frames.shape[1])
            frames_batch = frames[:, low: high]

            interpolated_frames = torch.vmap(
                torch.nn.functional.interpolate, in_dims=(1,), out_dims=(1,)
            )(frames_batch, size=desired_size, mode="bilinear", align_corners=False)

            x = self.Conv3D_A(interpolated_frames.transpose(1, 2))
            x = f.relu(x)
            x = self.Conv3D_B(x)
            x = f.relu(x)
            x = self.Conv3D_C(x)

            reconstructed_batch = torch.vmap(
                torch.nn.functional.interpolate, in_dims=(1,), out_dims=(1,)
            )(x.transpose(1, 2), size=desired_size, mode="bilinear", align_corners=False) + interpolated_frames
            reconstructed_video.append(reconstructed_batch)
        reconstructed_video = torch.concat(reconstructed_video, dim=1)

        loss = {}
        if self.training:
            loss["MAE"] = torch.mean(torch.abs(reconstructed_video - labels))

        return VSRModelOutput(
            reconstruction=reconstructed_video,
            loss=loss
        )
