import torch
from torch import Tensor
from torch.nn import Conv2d

from models.VSRModel import VSRModelOutput


class DummyBilinear(torch.nn.Module):
    def __init__(self):
        super(DummyBilinear, self).__init__()
        self.dummy = Conv2d(1, 1, 1)

    def forward(self, frames: Tensor, labels: Tensor) -> VSRModelOutput:
        desired_size = (2 * frames.shape[2], 2 * frames.shape[3])
        reconstruction = torch.nn.functional.interpolate(frames, size=desired_size,
                                                         mode="bilinear", align_corners=False)

        loss = {}
        if self.training:
            loss["MAE"] = torch.mean(torch.abs(reconstruction - labels))

        return VSRModelOutput(
            reconstruction=reconstruction,
            loss=loss
        )
