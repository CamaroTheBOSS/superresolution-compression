import torch
from torch import Tensor
from torch.nn import Conv2d

from models.VSRModel import VSRModelOutput


class DummyBilinearSR(torch.nn.Module):
    def __init__(self):
        super(DummyBilinearSR, self).__init__()
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


class DummyBilinearVSR(torch.nn.Module):
    def __init__(self):
        super(DummyBilinearVSR, self).__init__()
        self.dummy = Conv2d(1, 1, 1)

    def forward(self, frames: Tensor, labels: Tensor) -> VSRModelOutput:
        B, N, C, H, W = frames.shape
        desired_size = (2 * H, 2 * W) if labels is None else (labels.shape[-2], labels.shape[-1])
        reconstruction = torch.vmap(
            torch.nn.functional.interpolate, in_dims=(1,), out_dims=(1,)
        )(frames, size=desired_size, mode="bilinear", align_corners=False)

        loss = {}
        if self.training:
            loss["MAE"] = torch.mean(torch.abs(reconstruction - labels))

        return VSRModelOutput(
            reconstruction=reconstruction,
            loss=loss
        )
