from collections import namedtuple
from enum import Enum

import torch
from torch import Tensor
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from utils.utils import get_size_without_zero_padding


def psnr(input_images: Tensor, target_images: Tensor):
    psnr_values = []
    for i in range(target_images.size()[0]):
        x1, y1, x2, y2 = get_size_without_zero_padding(target_images[i], padding_value=-1)
        psnr_values.append(peak_signal_noise_ratio(input_images[i][:, y1:y2, x1:x2],
                                                   target_images[i][:, y1:y2, x1:x2], 1.0))

    return torch.sum(torch.stack(psnr_values))


def ssim(input_images: Tensor, target_images: Tensor):
    ssim_values = []
    for i in range(target_images.size()[0]):
        x1, y1, x2, y2 = get_size_without_zero_padding(target_images[i], padding_value=-1)
        ssim_values.append(structural_similarity_index_measure(input_images[i][:, y1:y2, x1:x2].unsqueeze(0),
                                                               target_images[i][:, y1:y2, x1:x2].unsqueeze(0)))
    return torch.sum(torch.stack(ssim_values))


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

        interpolated = torch.nn.functional.interpolate(target_images, size=[s // 2 for s in target_images.shape[-2:]],
                                                       mode="bilinear", align_corners=False)
        interpolated = torch.nn.functional.interpolate(interpolated, size=target_images.shape[-2:], mode="bilinear",
                                                       align_corners=False)
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
