from collections import namedtuple
from enum import Enum

import torch
from torch import Tensor
from utils.metrics import psnr, ssim
from utils.utils import interpolate


class Metric(Enum):
    _Metric = namedtuple("_Metric", "name, f")
    psnr = _Metric("psnr", psnr)
    ssim = _Metric("ssim", ssim)


class MetricAccumulator:
    @torch.no_grad()
    def __init__(self, metrics: list[Metric]):
        self.metrics = {metric.value.name: 0 for metric in metrics}
        self.functions = {metric.value.name: metric.value.f for metric in metrics}
        self.img_to_save = {"SR": [], "HR": [], "BILINEAR": []}

    @torch.no_grad()
    def update(self, input_images: Tensor, target_images: Tensor):
        for key in self.metrics:
            self.metrics[key] += self.functions[key](input_images, target_images)

        interpolated = interpolate(target_images, size=[s // 2 for s in target_images.shape[-2:]],
                                   mode="bilinear", align_corners=False)
        interpolated = interpolate(interpolated, size=target_images.shape[-2:], mode="bilinear", align_corners=False)
        self.img_to_save["SR"].append(input_images.cpu())
        self.img_to_save["HR"].append(target_images.cpu())
        self.img_to_save["BILINEAR"].append(interpolated.cpu())

    @torch.no_grad()
    def accumulate(self, divider: int):
        for key, value in self.metrics.items():
            self.metrics[key] = value / divider

    @torch.no_grad()
    def get_metrics(self):
        return self.metrics, self.img_to_save
