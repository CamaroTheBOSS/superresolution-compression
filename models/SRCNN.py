import torch
import torch.nn.functional as f
from torch import Tensor

from models.VSRModel import VSRModelOutput


class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.Conv2D_A = torch.nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.Conv2D_B = torch.nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.Conv2D_C = torch.nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, frames: Tensor, labels: Tensor) -> VSRModelOutput:
        desired_size = (2 * frames.shape[2], 2 * frames.shape[3])
        interpolated_frames = torch.nn.functional.interpolate(frames, size=desired_size,
                                                              mode="bilinear", align_corners=False)

        out = self.Conv2D_A(interpolated_frames)
        out = f.relu(out)
        out = self.Conv2D_B(out)
        out = f.relu(out)
        out = self.Conv2D_C(out)
        reconstruction = torch.nn.functional.interpolate(out, size=desired_size, mode="bilinear", align_corners=False)
        reconstruction = reconstruction + interpolated_frames

        loss = {}
        if self.training:
            loss["MAE"] = torch.mean(torch.abs(reconstruction - labels))

        return VSRModelOutput(
            reconstruction=reconstruction,
            loss=loss
        )

