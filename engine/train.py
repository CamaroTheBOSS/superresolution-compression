from collections import defaultdict

import torch
from torch import autocast
from torch.utils.data import DataLoader

from utils.logs import MetricLogger


def train_epoch(dataloader: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                lr_scheduler, epoch: int):
    def create_tensor():
        return torch.Tensor([0]).to(device)

    model.train()
    step_counter = 0
    metric_logger = MetricLogger("   ")
    loss_avr = defaultdict(create_tensor)
    for idx, batch in enumerate(metric_logger(dataloader, header=f"Epoch [{epoch}]")):
        step_counter += 1
        torch.cuda.empty_cache()
        frames = batch["LR"].to(device)
        frames_sr = batch["HR"].to(device)

        # forward pass
        outputs = model(frames, labels=frames_sr)
        loss = {key: value for key, value in outputs.loss.items()}
        loss["epoch"] = sum(loss_value for loss_value in outputs.loss.values())
        loss_avr = {key: loss_avr[key] + value for key, value in loss.items()}
        metric_logger.update(loss, prefix="loss_")
        metric_logger.update({"lr": optimizer.param_groups[0]['lr']})

        loss["epoch"].backward()
        optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        lr_scheduler.step(epoch + (step_counter / len(dataloader)))

    loss_avr = {key: value / len(dataloader) for key, value in loss_avr.items()}
    return loss_avr
