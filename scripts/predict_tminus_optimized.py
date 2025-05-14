import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from .arcane2.data.utils import (
    create_or_load_datamodule,
    create_or_load_testmodule_tminus_optimized,
    get_best_model_path,
)


@hydra.main(config_path="../config")
def main(cfg):
    """
    Main function for testing the model.
    """

    seed_everything(42, workers=True)

    device = cfg.get("device", "cpu")

    # Access the overrides from HydraConfig
    hydra_cfg = HydraConfig.get()

    # Find the exact override for boundaries in the command line
    for override in hydra_cfg.overrides.task:
        if override.startswith("+boundaries="):
            boundaries_name = override.split("=")[1]
            year = boundaries_name.split("_")[2]
            fold = boundaries_name.split("_")[3]

            break

    modelpath = None
    checkpoint_dir = (
        Path(hydra.utils.get_original_cwd() + cfg["training"]["checkpoint_dir"])
        / cfg["run_name"]
    )

    if "threshold" not in cfg["run_name"]:

        modelpath = get_best_model_path(
            checkpoint_dir,
            f"{cfg['run_name']}_{year}_{fold}",
        )
        print(f"Model path: {modelpath}")

    if modelpath and not modelpath.exists():
        print(f"Model file {modelpath} does not exist. Terminating.")
        return None

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    cache_path = (
        Path(hydra.utils.get_original_cwd() + cfg["cache_folder"]) / cfg["run_name"]
    )

    data_module = create_or_load_datamodule(
        cache_path,
        cfg["data_module"],
        cfg.get("force_load", False),
    )

    module = instantiate(cfg["module"])
    checkpoint = torch.load(modelpath, map_location=device, weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])

    modelname = f"{cfg['run_name']}_{fold}"

    # Conditional logging with Wandb
    if cfg.get("log_wandb", False):
        logger = WandbLogger(project=cfg["project"], name=cfg["run_name"])
        logger.watch(module, log="all")
    else:
        logger = None
        print("Wandb logging is disabled, running locally.")

    # Find the exact override for boundaries in the command line
    for override in hydra_cfg.overrides.task:
        if override.startswith("+base_dataset="):
            diff_name = override.split("=")[1]

            break

    diff_name_numbered = diff_name + "_tminus_all"

    test_module, load_cache = create_or_load_testmodule_tminus_optimized(
        cache_path,
        cfg.get("force_load", False),
        module,
        data_module,
        modelname,
        device=device,
        diff_name=diff_name_numbered,
        max_timestep=100,
    )

    if load_cache:
        test_module.run_inference_all_timesteps(
            modelname=modelname,
            max_timestep=100,
        )
        test_module.save(Path(cache_path), diff_name=diff_name_numbered)

    os._exit(0)  # Added this line to prevent the script from getting stuck


if __name__ == "__main__":
    main()
