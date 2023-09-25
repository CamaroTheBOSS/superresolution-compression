from collections import namedtuple
from enum import Enum

import torch
from torch import Tensor
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure


def psnr(input_images: Tensor, target_images: Tensor):
    psnr_values = []
    for i in range(target_images.size()[0]):
        psnr_values.append(peak_signal_noise_ratio(input_images[i], target_images[i], 1.0))

    return torch.sum(torch.stack(psnr_values))


def ssim(input_images: Tensor, target_images: Tensor):
    ssim_values = structural_similarity_index_measure(input_images, target_images)
    return ssim_values


class Metric(Enum):
    _Metric = namedtuple("_Metric", "name, f")
    psnr = _Metric("psnr", psnr)
    ssim = _Metric("ssim", ssim)


class MetricAccumulator:
    @torch.no_grad()
    def __init__(self, metrics: list[Metric]):
        self.metrics = {metric.value.name: 0 for metric in metrics}
        self.functions = {metric.value.name: metric.value.f for metric in metrics}
        self.img_to_save = {"SR": [], "HR": []}

    @torch.no_grad()
    def update(self, input_images, target_images):
        for key in self.metrics:
            self.metrics[key] += self.functions[key](input_images, target_images)

        self.img_to_save["SR"].append(input_images)
        self.img_to_save["HR"].append(target_images)

    @torch.no_grad()
    def accumulate(self, divider: int):
        for key, value in self.metrics.items():
            self.metrics[key] = value / divider

    @torch.no_grad()
    def get_metrics(self):
        return self.metrics, self.img_to_save
