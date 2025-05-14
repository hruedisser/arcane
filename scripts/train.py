import os
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from .arcane2.data.utils import create_or_load_datamodule


@hydra.main(config_path="../config")
def main(cfg):
    """
    Main function for training the model.
    """

    seed_everything(42, workers=True)

    device = cfg.get("device", "cpu")

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        prog_bar = True  # False if torch.cuda.is_available() else True

    else:
        device = torch.device(device)
        prog_bar = True

    print(f"Using device: {device}")

    # Access the overrides from HydraConfig
    hydra_cfg = HydraConfig.get()

    # Find the exact override for boundaries in the command line
    for override in hydra_cfg.overrides.task:
        if override.startswith("+boundaries="):
            boundaries_name = override.split("=")[1]

            year = boundaries_name.split("_")[2]
            fold = boundaries_name.split("_")[3]

            break

    cache_path = (
        Path(hydra.utils.get_original_cwd() + cfg["cache_folder"]) / cfg["run_name"]
    )

    data_module = create_or_load_datamodule(
        cache_path, cfg["data_module"], cfg.get("force_load", False)
    )

    print(data_module)
    print("Number of train samples: ")
    print(data_module.train_dataloader().sampler.num_samples)

    module = instantiate(cfg["module"])

    print(module)

    modelname = f"{cfg['run_name']}_{year}_{fold}"

    # Define the directory for saving checkpoints, including the run name
    checkpoint_dir = (
        Path(hydra.utils.get_original_cwd() + cfg["training"]["checkpoint_dir"])
        / cfg["run_name"]
    )

    # Ensure the directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Conditional logging with Wandb
    if cfg.get("log_wandb", False):
        logger = WandbLogger(project=cfg["project"], name=modelname)
        logger.watch(module, log="all")
        wandb.init(
            project="arcane-spiro",
            name=modelname,
        )
        wandb.run.name = modelname
    else:
        logger = None
        print("Wandb logging is disabled, running locally.")

    if prog_bar == False:
        print("Suppressing progress bar to avoid clutter.")

    # Log the checkpoint directory
    print(f"Trained models will be saved in: {checkpoint_dir.resolve()}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{modelname}" + "_{epoch}_{val_loss:.2f}",
        monitor=cfg["training"]["checkpoint_loss"],
        save_last=False,
        save_top_k=2,
        mode=cfg["training"]["checkpoint_mode"],
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",  # Metric to monitor
        min_delta=0.00,  # Minimum change to qualify as an improvement
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,  # Whether to print logs to stdout
        mode=cfg["training"][
            "checkpoint_mode"
        ],  # Minimize the monitored metric (val_loss)
    )

    additional_callbacks = []
    if "callbacks" in cfg["training"]:
        for callback in cfg["training"]["callbacks"]:
            additional_callbacks.append(instantiate(callback))

    # Determine devices to use based on config
    devices = cfg.get("num_gpus", 1)
    if devices == 1:
        devices = [0]
    elif devices == 2:
        devices = [1]

    trainer = Trainer(
        accumulate_grad_batches=cfg["training"].get("grad_batches", 1),
        callbacks=[checkpoint_cb, early_stop_callback] + additional_callbacks,
        fast_dev_run=cfg["training"].get("fast_dev_run", False),
        logger=logger,
        # precision=cfg["training"].get("precision", 32),
        max_epochs=cfg["training"].get("epochs", 1000),
        # profiler=cfg["training"].get("profiler", "simple"),
        # strategy=cfg["training"].get("strategy", "ddp"),
        devices=devices,
        accelerator=device,
        enable_progress_bar=prog_bar,
    )

    trainer.fit(module, data_module)

    os._exit(0)  # Added this line to prevent the script from getting stuck


if __name__ == "__main__":
    main()
