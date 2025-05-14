import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from ..data_utils.conversion import convert_to_timestamp
from .dataset_base import DatasetBase
from .sequential_dataset import SequentialDataset


class BoundaryFilteredDataset(DatasetBase):
    def __init__(
        self,
        dataset: DatasetBase,
        boundaries: List[Tuple[str, str]],
    ):
        super().__init__()

        self.boundaries = boundaries

        print("Initializing BoundaryFilteredDataset.")
        print(f"Boundaries: {self.boundaries}")

        # Convert boundaries to numpy.datetime64 using the helper function
        boundaries_dt = [
            (
                convert_to_timestamp(b[0]),
                convert_to_timestamp(b[1]),
            )
            for b in self.boundaries
        ]

        # Sort boundaries by start time
        boundaries_dt = sorted(boundaries_dt, key=lambda x: x[0])

        logger.info(f"Sorted boundaries: {boundaries_dt}")

        # Check if boundaries are adjacent and merge if necessary
        for i in range(len(boundaries_dt) - 1):
            if boundaries_dt[i][1] == boundaries_dt[i + 1][0]:
                boundaries_dt[i] = (boundaries_dt[i][0], boundaries_dt[i + 1][1])
                boundaries_dt.pop(i + 1)
        logger.info(f"Merged boundaries: {boundaries_dt}")

        # Check if dataset is a SequentialDataset and adjust boundaries accordingly
        if isinstance(dataset, SequentialDataset):
            logger.info(f"Original boundaries: {boundaries_dt}")
            boundaries_dt = self.adjust_boundaries(
                boundaries_dt,
                dataset.n_samples,
                dataset.skip_n,
                dataset.max_time_gap,
            )
            logger.info(f"Adjusted boundaries: {boundaries_dt}")

        # Use the dataset timestamps directly
        timestamps = dataset.timestamps

        indices = []
        for b0, b1 in boundaries_dt:
            # Find indices where timestamps are within the boundary
            idxs = [i for i in tqdm(range(len(timestamps))) if b0 < timestamps[i] < b1]
            indices += idxs

        self.fwd_indices = {i: idx for i, idx in enumerate(indices)}
        self.bwd_indices = {idx: i for i, idx in enumerate(indices)}

        self.dataset = dataset
        self.weights = np.array(dataset.weights)[indices]

    @property
    def id(self):
        return self.dataset.id

    def __len__(self):
        return len(self.fwd_indices)

    def get_data(self, idx):
        return self.dataset.get_data(self.fwd_indices[idx])

    def get_timestamp(self, idx):
        return self.dataset.get_timestamp(self.fwd_indices[idx])

    def get_timestamp_idx(self, timestamp):
        return self.bwd_indices[self.dataset.get_timestamp_idx(timestamp)]

    @property
    def sensor_ids(self):
        return self.dataset.sensor_ids

    def __repr__(self):
        inner_repr = repr(self.dataset)
        lines = inner_repr.split("\n")
        inner_repr = "\n".join(["\t" + line for line in lines])

        boundaries = "\n".join([f"\t{b[0]} - {b[1]}" for b in self.boundaries])

        return (
            f"BoundaryFilteredDataset - {len(self)} samples\n{inner_repr}\n{boundaries}"
        )

    def adjust_boundaries(self, boundaries_dt, n_samples, skip_n, max_time_gap):

        boundaries = [
            (
                b0,
                int(b1 - (n_samples * (skip_n + 1) + max_time_gap)),
            )
            for b0, b1 in boundaries_dt
        ]

        return boundaries

    def save(self, path, current_cfg, overwrite=True):
        root_path = Path(path)
        if root_path.exists() and not overwrite:
            raise IOError(f"File {path} already exists and not overwriting")

        data_path = root_path / "dataset.pkl"

        with open(data_path, "wb") as path:
            pickle.dump(self.dataset, path)

        config_cache_path = root_path / "config.pkl"
        with open(config_cache_path, "wb") as path:
            pickle.dump(current_cfg, path)

    @staticmethod
    def load(
        root_path,
        boundaries,
    ):
        root_path = Path(root_path)

        data_path = root_path / "dataset.pkl"

        with open(data_path, "rb") as path:
            dataset = pickle.load(path)

        return BoundaryFilteredDataset(
            dataset,
            boundaries,
        )

    @staticmethod
    def check_load_cache(root_path, current_cfg):
        root_path = Path(root_path)
        data_path = root_path / "dataset.pkl"

        if not data_path.exists():
            logger.info(f"Cache path {data_path} does not exist.")
            return False

        with open(root_path / "config.pkl", "rb") as path:
            cached_cfg = pickle.load(path)
            if current_cfg != cached_cfg:
                logger.info("Configurations do not match.")
                return False

        return True
