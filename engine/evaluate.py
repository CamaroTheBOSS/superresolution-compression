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

        frames = batch["LR"].to(device)
        frames_sr = batch["HR"].to(device)

        with torch.no_grad():
            outputs = model(frames, labels=frames_sr)
            metric_accumulator.update(outputs.reconstruction, frames_sr)

    metric_accumulator.accumulate((len(dataloader) * dataloader.batch_size))
    return metric_accumulator.get_metrics()
