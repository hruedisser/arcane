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

from .arcane2.data.abstract.boundary_filtered_dataset import BoundaryFilteredDataset
from .arcane2.data.datamodule import ParsedDataModule
from .arcane2.data.utils import create_group_boundaries, get_sampler_args


@hydra.main(config_path="../config")
def main(cfg):
    """
    Main function for training the model.
    """

    seed_everything(42, workers=True)

    device = cfg.get("device", "cpu")

    print("Trying to use device: ", device)

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

    cache_path = (
        Path(hydra.utils.get_original_cwd() + cfg["cache_folder"]) / cfg["run_name"]
    )

    # manually preparing boundaries for training

    years_rtsw = [1998 + i for i in range(27)]
    test_years = [2023, 2024, 2025]

    remaining_years = [year for year in years_rtsw if year not in test_years]

    if len(remaining_years) % 3 == 2:
        group_size = (len(remaining_years) // 3) + 1
    else:
        group_size = len(remaining_years) // 3

    group1 = remaining_years[:group_size]
    group2 = remaining_years[group_size : 2 * group_size]
    group3 = remaining_years[2 * group_size :]

    test_boundaries = [[f"{test_years[0]}0101T000000", f"{test_years[-1]}0101T000000"]]

    for n in range(3):
        print(f"Training fold {n}")

        if n == 0:
            train_years = group1 + group2
            val_years = group3
        elif n == 1:
            train_years = group1 + group3
            val_years = group2
        elif n == 2:
            train_years = group2 + group3
            val_years = group1
        else:
            raise ValueError("n must be 0, 1, or 2")

        train_boundaries = create_group_boundaries(train_years)
        val_boundaries = create_group_boundaries(val_years)

        print(f"Train boundaries: {train_boundaries}")
        print(f"Val boundaries: {val_boundaries}")
        print(f"Test boundaries: {test_boundaries}")

        cfg_datamodule = cfg["data_module"]

        cache_path = Path(cache_path)

        print(f"Current working directory: {Path.cwd()}")
        print(f"Checking cache path: {cache_path}")
        cache_path.mkdir(parents=True, exist_ok=True)

        load_cache = False
        if BoundaryFilteredDataset.check_load_cache(cache_path, cfg_datamodule):
            load_cache = True
        if load_cache == False:
            print("Cache not found, instantiating datamodule.")
            train_dataset = instantiate(cfg_datamodule["train_dataset"])
            print(f"Saving dataset to {cache_path}")
            train_dataset.save(cache_path, cfg_datamodule)

        print(f"Loading datamodule from {cache_path}")

        test_dataset = BoundaryFilteredDataset.load(cache_path, test_boundaries)

        train_dataset = BoundaryFilteredDataset.load(cache_path, train_boundaries)

        val_dataset = BoundaryFilteredDataset.load(cache_path, val_boundaries)

        data_module = ParsedDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=cfg_datamodule["batch_size"],
            num_workers=cfg_datamodule["num_workers"],
            shuffle=cfg_datamodule["shuffle"],
            # Train sampler logic
            train_sampler=(
                instantiate(
                    cfg_datamodule["train_sampler"],
                    **get_sampler_args(cfg_datamodule["train_sampler"], train_dataset),
                )
                if cfg_datamodule["train_sampler"]
                else None
            ),
            # Validation sampler logic
            val_sampler=(
                instantiate(
                    cfg_datamodule["val_sampler"],
                    **get_sampler_args(cfg_datamodule["val_sampler"], val_dataset),
                )
                if cfg_datamodule["val_sampler"]
                else None
            ),
            # Test sampler logic
            test_sampler=(
                instantiate(
                    cfg_datamodule["test_sampler"],
                    **get_sampler_args(cfg_datamodule["test_sampler"], test_dataset),
                )
                if cfg_datamodule["test_sampler"]
                else None
            ),
            # Collate functions
            train_collate_fn=instantiate(cfg_datamodule["train_collate_fn"]),
            val_collate_fn=instantiate(cfg_datamodule["val_collate_fn"]),
            test_collate_fn=instantiate(cfg_datamodule["test_collate_fn"]),
        )

        print(data_module)
        print("Number of train samples: ")
        print(data_module.train_dataloader().sampler.num_samples)

        module = instantiate(cfg["module"])

        print(module)

        modelname = f"{cfg['run_name']}_{n}"

        checkpoint_dir = cache_path

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
            save_top_k=1,
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

        if cfg.get("device", "cpu") == "cpu":
            devices = 1
            accelerator = "cpu"
        else:
            accelerator = device

        trainer = Trainer(
            accumulate_grad_batches=cfg["training"].get("grad_batches", 1),
            callbacks=[checkpoint_cb, early_stop_callback] + additional_callbacks,
            fast_dev_run=cfg["training"].get("fast_dev_run", False),
            logger=logger,
            # precision=cfg["training"].get("precision", 32),
            max_epochs=cfg["training"].get("epochs", 1000),
            # profiler=cfg["training"].get("profiler", "simple"),
            # strategy=cfg["training"].get("strategy", "ddp"),
            # devices=devices,
            accelerator=accelerator,
            enable_progress_bar=prog_bar,
        )

        trainer.fit(module, data_module)

    os._exit(0)  # Added this line to prevent the script from getting stuck


if __name__ == "__main__":
    main()
