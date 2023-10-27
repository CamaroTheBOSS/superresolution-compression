import torch
from torch import autocast
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from engine.MetricAccumulator import MetricAccumulator, Metric


def evaluate_model(dataloader: DataLoader, model: torch.nn.Module, device: torch.device) -> dict:
    model.eval()
    metric_accumulator = MetricAccumulator([Metric.psnr, Metric.ssim])

    for idx, batch in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()

        lr_frames = batch["LR"].to(device)
        hr_frames = batch["HR"].to(device)

        with torch.no_grad():
            outputs = model(lr_frames, labels=hr_frames)
            metric_accumulator.update(outputs.reconstruction, hr_frames)
    metric_accumulator.accumulate(len(dataloader.dataset))
    return metric_accumulator.get_metrics()
