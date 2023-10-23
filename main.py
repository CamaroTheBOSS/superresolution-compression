import sys
import traceback

import typer
import wandb
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from engine.evaluate import evaluate_model
from engine.train import train_epoch
from models.DummyBilinear import DummyBilinear
from models.SRCNN import SRCNN
from models.VSRDataset import VSRDataset, collate_fn, SRDataset
from models.VSRModel import VSRModel
from utils.logs import FileLogger, log_to_wandb
from utils.utils import init_training_directory, save_args, save_results

app = typer.Typer()


@app.command("train")
def train(name: str = typer.Option("run", "--name", help="Name of run"),
          group_name: str = typer.Option("", "--group-name", help="Name of group in wandb"),
          epochs: int = typer.Option(30, "--epochs", help="Number of epochs in training"),
          batch: int = typer.Option(8, "--batch", help="Batch size used in training"),
          learning_rate: float = typer.Option(3e-4, "--lr", help="Initial training learning rate"),
          output_root: str = typer.Option("./outputs", "--output-dir", help="Output results root directory"),
          disable_wandb: bool = typer.Option(False, "--disable-wandb", help="Whether to use wandb")) -> None:
    if not disable_wandb:
        wandb.login(key="412dfebd88d6fb0be9f1caf8fd73c2363c48c8e0")
        wandb.init(project="superresolution-compression", name=name, group=group_name)

    torch.manual_seed(107)
    output_dir = init_training_directory(output_root, name)
    logger = FileLogger(output_dir, "train.log")
    save_args(output_dir, **dict(name=name, group_name=group_name, epochs=epochs, batch=batch, learning_rate=learning_rate,
                                 output_root=output_root, disable_wandb=disable_wandb))

    src_dataset = './datasets/SR'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    # 2. Initialize dataloaders
    # train_set = VSRDataset(src_dataset)
    # test_set = VSRDataset(src_dataset, test=True)
    train_set = SRDataset(src_dataset, ["BSD100", "Set5", "Urban100"])
    test_set = SRDataset(src_dataset, ["Set14"])

    train_dataloader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=batch, shuffle=False, collate_fn=collate_fn)

    # 3. Get the model and set the optimizer
    model = SRCNN().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    steps = epochs + 10
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=steps, T_mult=1)
    try:
        for epoch in range(epochs):
            loss_dict = train_epoch(train_dataloader, model, optimizer, device, lr_scheduler, epoch)
            metric_dict, img_to_save = evaluate_model(test_dataloader, model, device)

            save_results(img_to_save, f"{output_dir}/epoch_{epoch + 1}")

            if not disable_wandb:
                log_to_wandb(loss_dict, metric_dict)

            if (epoch + 1) % 50 == 0:
                torch.save({'model_state_dict': model.state_dict(), 'model_name': "superresolution"},
                           f"{output_dir}/model_{epoch + 1}.pth")
    except Exception:
        traceback.print_exc(file=sys.stdout)
        _, exc_value, tb = sys.exc_info()
        logger.log(traceback.TracebackException(type(exc_value), exc_value, tb, limit=None).format(chain=True))

    wandb.finish()


if __name__ == "__main__":
    app()
