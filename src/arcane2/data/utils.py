import re
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import instantiate
from loguru import logger

from ..modules.test_module_tminus import TestModuleTMinusOptimized
from .abstract.boundary_filtered_dataset import BoundaryFilteredDataset
from .data_utils.event import find, overlap_with_list
from .datamodule import ParsedDataModule

# We create a function to compare the extracted catalog to the original catalog


def compare_catalogs_for_results_all_FPs(
    extracted_cat, detectable_original_cat, thresh=0.01
):
    """
    Compare the extracted catalog to the original catalog and return the results.
    """

    TP = []
    FP = []
    FN = []
    detected = []
    delays = []
    durations = []

    if extracted_cat == []:
        print("No events detected.")
        FN = detectable_original_cat
    else:
        for origin_event in detectable_original_cat:
            overlaps = overlap_with_list(origin_event, extracted_cat, percent=True)

            overlaps_over_thresh = [
                extracted_cat[overlap_index]
                for overlap_index in range(len(overlaps))
                if overlaps[overlap_index] > thresh
            ]

            if overlaps_over_thresh == []:
                FN.append(origin_event)
            else:
                TP.append(overlaps_over_thresh[0])
                detected.append(origin_event)
                durations.append(
                    (origin_event.end - origin_event.begin).total_seconds() / 60
                )
                delays.append(
                    (overlaps_over_thresh[0].begin - origin_event.begin).total_seconds()
                    / 60
                )

    if detectable_original_cat == []:
        print("No events in the original catalog.")
        FP = extracted_cat
    else:
        FP = [x for x in extracted_cat if x not in TP]

    # round delays to 10 minutes
    delays = [round(delay / 10) * 10 for delay in delays]

    ious = [detect.intersection_over_union(x) for detect, x in zip(TP, detected)]

    return TP, FP, FN, delays, detected, durations, ious


def compare_catalogs_for_results_all_TPs(
    extracted_cat, detectable_original_cat, thresh=0.01
):
    """
    Compare the extracted catalog to the original catalog and return the results.
    """

    TP = []
    FP = []
    FN = []
    detected = []
    delays = []
    durations = []

    if extracted_cat == []:
        print("No events detected.")
        FN = detectable_original_cat
    else:
        for origin_event in detectable_original_cat:
            overlaps = overlap_with_list(origin_event, extracted_cat, percent=True)

            overlaps_over_thresh = [
                extracted_cat[overlap_index]
                for overlap_index in range(len(overlaps))
                if overlaps[overlap_index] > thresh
            ]

            if overlaps_over_thresh == []:
                FN.append(origin_event)
            else:
                for overlap_event in overlaps_over_thresh:
                    TP.append(overlap_event)
                    if origin_event not in detected:
                        detected.append(origin_event)
                    durations.append(
                        (origin_event.end - origin_event.begin).total_seconds() / 60
                    )
                    delays.append(
                        (overlap_event.begin - origin_event.begin).total_seconds() / 60
                    )

    if detectable_original_cat == []:
        print("No events in the original catalog.")
        FP = extracted_cat
    else:
        FP = [x for x in extracted_cat if x not in TP]

    # round delays to 10 minutes
    delays = [round(delay / 10) * 10 for delay in delays]

    ious = [detect.intersection_over_union(x) for detect, x in zip(TP, detected)]

    return TP, FP, FN, delays, detected, durations, ious


def compare_catalogs_for_results(extracted_cat, detectable_original_cat, thresh=0.01):
    """
    Compare the extracted catalog to the original catalog and return the results.
    """

    TP = []
    FP = []
    FN = []
    found_already = []
    detected = []
    delays = []
    durations = []

    if extracted_cat == []:
        print("No events detected.")
        FN = detectable_original_cat
    else:
        for origin_event in detectable_original_cat:
            corresponding = find(
                origin_event, extracted_cat, thresh=0.01, choice="first"
            )

            if corresponding is None:
                FN.append(origin_event)
            else:
                TP.append(corresponding)
                detected.append(origin_event)
                durations.append(
                    (origin_event.end - origin_event.begin).total_seconds() / 60
                )

                delays.append(
                    (corresponding.begin - origin_event.begin).total_seconds() / 60
                )
                found_already.append(corresponding)

    if detectable_original_cat == []:
        print("No events in the original catalog.")
        FP = extracted_cat
    else:
        FP = [
            x
            for x in extracted_cat
            if max(overlap_with_list(x, detectable_original_cat, percent=True)) == 0
        ]

    # round delays to 10 minutes
    delays = [round(delay / 10) * 10 for delay in delays]

    ious = [detect.intersection_over_union(x) for detect, x in zip(TP, detected)]

    return TP, FP, FN, delays, found_already, detected, durations, ious


def create_group_boundaries(years_group):
    if not years_group:
        return []
    boundaries = []
    start_year = years_group[0]
    end_year = years_group[0]
    for year in years_group[1:]:
        if year - end_year == 1:
            end_year = year
        else:
            boundaries.append([f"{start_year}0101T000000", f"{end_year+1}0101T000000"])
            start_year = year
            end_year = year
    boundaries.append([f"{start_year}0101T000000", f"{end_year+1}0101T000000"])
    return boundaries


def create_or_load_module(checkpoint_path: str | Path | None, cfg, device: str = "cpu"):
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            model = instantiate(cfg["model"])
            model = torch.load(checkpoint_path, map_location=device)

            model.num_classes = cfg["model"]["num_classes"]

            if model.__class__.__name__.contains("ResUNet"):
                module = instantiate(cfg["module"], model=model)
            else:
                raise ValueError(f"Unsupported model class: {model.__class__.__name__}")

        else:
            module = instantiate(cfg["module"])
    else:
        module = instantiate(cfg["module"])

    return module


def get_sampler_args(sampler_cfg, dataset):
    if sampler_cfg["_target_"] == "torch.utils.data.RandomSampler":
        logger.info(
            f"RandomSampler: {sampler_cfg.get('num_samples', len(dataset))} samples, replacement={sampler_cfg.get('replacement', False)}"
        )
        return {
            "data_source": dataset,
            "num_samples": sampler_cfg.get("num_samples", len(dataset)),
            "replacement": sampler_cfg.get("replacement", False),
        }

    elif sampler_cfg["_target_"] == "torch.utils.data.WeightedRandomSampler":
        logger.info(
            f"WeightedRandomSampler: {sampler_cfg.get("num_samples", len(dataset))} samples, replacement={sampler_cfg.get('replacement', False)}"
        )
        return {
            "weights": dataset.weights,
            "num_samples": sampler_cfg.get("num_samples", len(dataset)),
            "replacement": sampler_cfg.get("replacement", False),
        }

    elif sampler_cfg["_target_"] == "src.arcane2.data.samplers.CustomSubsetSampler":
        logger.info(
            f"CustomSubsetSampler: {sampler_cfg.get('every_n_items', 1)} samples"
        )
        return {
            "data_source": dataset,
            "every_n_items": sampler_cfg.get("every_n_items", 1),
        }


def get_best_model_path(model_dir, modelname, method="best"):
    """
    Find the model path with the lowest val_loss for the given model name.
    Parameters:
        model_dir (str or Path): Path to the directory containing model checkpoints.
        modelname (str): The base name of the model.

    Returns:
        Path: The path to the model with the lowest val_loss, or None if no valid model is found.
    """
    model_dir = Path(model_dir)
    logger.info(f"Searching for best model in {model_dir}")

    logger.info(f"Total models found: {len(list(model_dir.glob('*.ckpt')))}")

    pattern = re.compile(
        rf"{re.escape(modelname)}_epoch=(\d+)_val_loss=(\d+\.\d+)\.ckpt"
    )
    best_model_path = None
    lowest_val_loss = float("inf")
    highest_epoch = -1

    for model_file in model_dir.glob("*.ckpt"):
        match = pattern.match(model_file.name)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            if val_loss < lowest_val_loss or (
                val_loss == lowest_val_loss and epoch > highest_epoch
            ):
                lowest_val_loss = val_loss
                highest_epoch = epoch
                best_model_path = model_file

    logger.info(f"Best model: {best_model_path}")

    return best_model_path


def create_or_load_datamodule_force_rerun(
    cache_path: str | Path, cfg, no_ask: bool = False
):
    test_dataset = instantiate(cfg["test_dataset"])

    train_dataset = instantiate(cfg["train_dataset"])

    val_dataset = instantiate(cfg["val_dataset"])

    data_module = ParsedDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=cfg["shuffle"],
        # Train sampler logic
        train_sampler=(
            instantiate(
                cfg["train_sampler"],
                **get_sampler_args(cfg["train_sampler"], train_dataset),
            )
            if cfg["train_sampler"]
            else None
        ),
        # Validation sampler logic
        val_sampler=(
            instantiate(
                cfg["val_sampler"], **get_sampler_args(cfg["val_sampler"], val_dataset)
            )
            if cfg["val_sampler"]
            else None
        ),
        # Test sampler logic
        test_sampler=(
            instantiate(
                cfg["test_sampler"],
                **get_sampler_args(cfg["test_sampler"], test_dataset),
            )
            if cfg["test_sampler"]
            else None
        ),
        # Collate functions
        train_collate_fn=instantiate(cfg["train_collate_fn"]),
        val_collate_fn=instantiate(cfg["val_collate_fn"]),
        test_collate_fn=instantiate(cfg["test_collate_fn"]),
    )

    return data_module


def create_or_load_datamodule(cache_path: str | Path, cfg, no_ask: bool = False):
    cache_path = Path(cache_path)
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Checking cache at {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

    load_cache = False
    if BoundaryFilteredDataset.check_load_cache(cache_path, cfg):
        if not no_ask:
            load_cache = input(
                f"Found dataset cache at {cache_path}. Load from cache? ([yes]/no): "
            ) in ["", "yes"]
        else:
            load_cache = True
    if load_cache == False:
        logger.info("Cache not found or not loading from cache, instantiating dataset")
        train_dataset = instantiate(cfg["train_dataset"])
        print(f"Saving model in {cache_path}")

        train_dataset.save(cache_path, cfg)

    print(f"Loading data module from {cache_path}")
    test_dataset = BoundaryFilteredDataset.load(
        cache_path,
        cfg.get("test_dataset").boundaries,
    )

    train_dataset = BoundaryFilteredDataset.load(
        cache_path,
        cfg.get("train_dataset").boundaries,
    )

    val_dataset = BoundaryFilteredDataset.load(
        cache_path,
        cfg.get("val_dataset").boundaries,
    )

    data_module = ParsedDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=cfg["shuffle"],
        # Train sampler logic
        train_sampler=(
            instantiate(
                cfg["train_sampler"],
                **get_sampler_args(cfg["train_sampler"], train_dataset),
            )
            if cfg["train_sampler"]
            else None
        ),
        # Validation sampler logic
        val_sampler=(
            instantiate(
                cfg["val_sampler"], **get_sampler_args(cfg["val_sampler"], val_dataset)
            )
            if cfg["val_sampler"]
            else None
        ),
        # Test sampler logic
        test_sampler=(
            instantiate(
                cfg["test_sampler"],
                **get_sampler_args(cfg["test_sampler"], test_dataset),
            )
            if cfg["test_sampler"]
            else None
        ),
        # Collate functions
        train_collate_fn=instantiate(cfg["train_collate_fn"]),
        val_collate_fn=instantiate(cfg["val_collate_fn"]),
        test_collate_fn=instantiate(cfg["test_collate_fn"]),
    )

    return data_module


def create_or_load_testmodule_tminus_optimized(
    cache_path: str | Path,
    no_ask: bool = False,
    classifier_module=None,
    data_module=None,
    modelname=None,
    device="cpu",
    diff_name="",
    max_timestep=100,
):
    cache_path = Path(cache_path)
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Checking cache at {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

    if classifier_module == None or data_module == None:
        raise ValueError("classifier_module and test_dataloader must be provided")

    load_cache = False
    if TestModuleTMinusOptimized.check_load_cache(cache_path, diff_name=diff_name):
        if not no_ask:
            load_cache = input(
                f"Found testmodule cache at {cache_path}. Load from cache? ([yes]/no): "
            ) in ["", "yes"]
        else:
            load_cache = True

    if load_cache:
        print(f"Loading data module from {cache_path}")
        test_module = TestModuleTMinusOptimized.load(
            cache_path,
            classifier_module,
            data_module.test_dataloader(),
            device=device,
            diff_name=diff_name,
        )
    else:
        test_module = TestModuleTMinusOptimized(
            classifier_module,
            data_module.test_dataloader(),
            device=device,
        )
        df = data_module.test_dataset.dataset.dataset.df
        test_module.run_inference_all_timesteps(
            df=df,
            modelname=modelname,
            max_timestep=max_timestep,
        )

        print(f"Saving test_module in {cache_path}")
        test_module.save(cache_path, diff_name=diff_name)

    test_module.device = device
    return test_module, load_cache


def merge_columns_by_mean(
    df, prefix="predicted_value_train_arcane_rtsw_new_", tminus_range=range(1, 101)
):
    """
    Merge columns by taking the mean of the columns with the same prefix and tminus
    """
    merged_data = {}
    columns_to_remove = []
    for t in tminus_range:
        cols_to_merge = [f"{prefix}{i}_tminus{t}" for i in range(3)]
        if all(col in df.columns for col in cols_to_merge):
            merged_data[f"{prefix}tminus{t}"] = df[cols_to_merge].mean(axis=1)
            columns_to_remove.extend(cols_to_merge)

    merged_df = pd.DataFrame(merged_data, index=df.index)

    df = df.drop(columns=columns_to_remove)

    df = pd.concat([df, merged_df], axis=1)

    return df


def merge_columns_by_mean_kp(df, prefix="keyparam_pred_0_train_cumsum_"):
    """
    Merge columns by taking the mean of the columns with the same prefix
    """
    merged_data = {}
    columns_to_remove = []
    cols_to_merge = [f"{prefix}{i}" for i in range(3)]
    if all(col in df.columns for col in cols_to_merge):
        merged_data[f"{prefix}"] = df[cols_to_merge].mean(axis=1)
        columns_to_remove.extend(cols_to_merge)

    merged_df = pd.DataFrame(merged_data, index=df.index)

    df = df.drop(columns=columns_to_remove)

    df = pd.concat([df, merged_df], axis=1)

    return df


def shift_columns(df):
    for col in df.columns:
        if "_tminus" in col:
            # Extract the time shift value (X)
            shift_value = int(col.split("_tminus")[-1])
            # Shift the column by the shift_value
            df[col] = df[col].shift(-shift_value)  # shift forward by negative value
    return df
