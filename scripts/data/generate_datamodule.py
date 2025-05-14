from pathlib import Path

import hydra

from arcane2.data.utils import create_or_load_datamodule


@hydra.main(config_path="../../config")
def main(cfg):

    cache_path = Path(cfg["cache_folder"]) / cfg["run_name"]
    data_module = create_or_load_datamodule(cache_path, cfg["data_module"])
    print(data_module)


if __name__ == "__main__":
    main()
