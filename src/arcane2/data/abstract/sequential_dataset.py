from typing import Any, Dict, List, Literal

import numpy as np
from loguru import logger
from tqdm import tqdm

from .dataset_base import DatasetBase


class SequentialDataset(DatasetBase):
    def __init__(
        self,
        dataset: DatasetBase,
        n_samples=2,
        skip_n=0,
        stride=1,
        max_time_gap: float = 10.0,
        timestamp_idx: Literal["first", "last"] = "last",
        return_timestamps=False,
        filters: bool = False,
        filter_type: Literal["all", "half"] = "half",
        filter_key: str = "NGUYEN_catalog-ICME",
        weights: bool = False,
        weight_type: Literal["binary", "half-percentage"] = "binary",
        weight_factor: float = 10.0,
        chunk_size: int = 10000,
    ):
        super().__init__()

        self.dataset = dataset
        self.n_samples = n_samples
        self.skip_n = skip_n
        self.stride = stride
        self.max_time_gap = max_time_gap
        self.timestamp_idx = timestamp_idx
        self.return_timestamps = return_timestamps
        self.chunk_size = chunk_size

        self.idx_format = tuple(
            [0] + [i * (1 + self.skip_n) for i in range(1, n_samples)]
        )

        # Apply filtering if enabled
        self.idxs = (
            self._filter_valid_idxs(filter_type, filter_key)
            if filters
            else self._compute_valid_idxs()
        )

        self.weights = (
            self._compute_weights(weight_type, weight_factor, filter_key)
            if weights
            else np.ones(len(self.idxs))
        )

    def _compute_valid_idxs(self) -> List[int]:
        """Compute the list of valid indices based on the stride."""

        if self.max_time_gap > 0:
            logger.info(
                f"Computing valid indices with max time gap of {self.max_time_gap}"
            )

            n_rows = len(self.dataset.df)

            # Create valid indices according to the stride
            valid_idxs = np.arange(0, (n_rows - self.n_samples), self.stride)

            filtered_idxs = []

            for chunk_start in tqdm(range(0, len(valid_idxs), self.chunk_size)):
                chunk_end = min(chunk_start + self.chunk_size, len(valid_idxs))
                idx_chunk = valid_idxs[chunk_start:chunk_end]

                # Compute the sequence indices for this chunk
                seq_idxs_matrix = np.array(
                    [idx_chunk + i * (1 + self.skip_n) for i in range(self.n_samples)]
                ).T

                # Process each row independently to avoid large memory consumption
                for row_idxs in seq_idxs_matrix:

                    time_diffs = (
                        self.dataset.df.iloc[row_idxs].index.diff().total_seconds() / 60
                    )

                    if any(time_diffs > self.max_time_gap):
                        continue

                    filtered_idxs.append(
                        row_idxs[0]
                    )  # Append the starting index of the valid sequence

            logger.info(
                f"Filtered out {len(valid_idxs) - len(filtered_idxs)} sequences"
            )

            return filtered_idxs

        else:
            return [
                i * self.stride
                for i in range((len(self.dataset) - self.n_samples) // self.stride)
            ]

    def _filter_valid_idxs(self, filter_type, filter_key) -> List[int]:
        """Filter out indices where 'catalog' exceeds the cut_percentile or meets the special condition."""

        logger.info(f"Filtering dataset based on {filter_key} values")

        n_rows = len(self.dataset.df)

        # Create valid indices according to the stride
        valid_idxs = np.arange(0, (n_rows - self.n_samples), self.stride)

        filtered_idxs = []

        for chunk_start in tqdm(range(0, len(valid_idxs), self.chunk_size)):
            chunk_end = min(chunk_start + self.chunk_size, len(valid_idxs))
            idx_chunk = valid_idxs[chunk_start:chunk_end]

            # Compute the sequence indices for this chunk
            seq_idxs_matrix = np.array(
                [idx_chunk + i * (1 + self.skip_n) for i in range(self.n_samples)]
            ).T

            # Process each row independently to avoid large memory consumption
            for row_idxs in seq_idxs_matrix:
                catalog_values = self.dataset.df.iloc[row_idxs][filter_key].values

                if filter_type == "half":
                    # Only consider values in the second half of the sequence
                    if any(catalog_values[self.n_samples // 2 :] == 2):
                        continue
                elif filter_type == "all":
                    # Consider the whole sequence
                    if any(catalog_values == 2):
                        continue

                filtered_idxs.append(
                    row_idxs[0]
                )  # Append the starting index of the valid sequence

        return filtered_idxs

    def _compute_weights(self, weight_type, weight_factor, filter_key) -> np.ndarray:
        """Compute the weights for each sample based on the 'weight_type'."""
        if weight_type == "binary":

            logger.info(f"Computing binary weights based on {filter_key} values")

            valid_idxs = np.array(self.idxs)

            # Calculate the last index for each sequence
            last_idxs = valid_idxs + (self.n_samples - 1) * (1 + self.skip_n)

            # Fetch the catalog values for the last indices in a vectorized manner
            catalog_values_last = self.dataset.df.iloc[last_idxs][filter_key].values

            # Initialize the weights to ones
            weights = np.ones(len(self.idxs))

            # Set the weight to weight_factor where the last catalog value is 1
            weights[catalog_values_last == 1] = weight_factor

            return weights

        elif weight_type == "half-percentage":
            raise NotImplementedError("Half-percentage weights not implemented yet")

        elif weight_type == "None":
            return np.ones(len(self.idxs))

    def __len__(self) -> int:
        return len(self.idxs)

    @property
    def timestamps(self) -> List[int]:
        return [self.dataset.get_timestamp(i) for i in self.idxs]

    def get_data(self, idx) -> Dict[str, Any]:
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        original_idx = self.idxs[idx]
        seq_idxs = [original_idx + f for f in self.idx_format]

        data_list = [self.dataset.get_data(i) for i in seq_idxs]
        data = {d: [] for d in data_list[0]}
        for d in data_list:
            for key in d:
                data[key].append(d[key])

        # Convert lists to numpy arrays
        for key in data:
            data[key] = np.array(data[key])

        if self.return_timestamps:
            seq_timestamps = [self.dataset.get_timestamp(i) for i in seq_idxs]
            data["timestamps"] = np.array(seq_timestamps)

        return data

    @property
    def sensor_ids(self):
        return self.dataset.sensor_ids

    @property
    def id(self):
        return self.dataset.id

    @property
    def satellite_name(self):
        return self.dataset.satellite_name

    def __repr__(self) -> str:
        inner_repr = repr(self.dataset)
        lines = inner_repr.split("\n")
        inner_repr = "\n".join(["\t" + line for line in lines])
        return f"Sequential - {len(self.idxs)} samples\n{inner_repr}"

    def get_timestamp(self, idx):
        original_idx = self.idxs[idx]
        if self.timestamp_idx == "first":
            return self.dataset.get_timestamp(original_idx)
        else:
            return self.dataset.get_timestamp(original_idx + self.idx_format[-1])

    def get_timestamp_idx(self, timestamp):
        return self.timestamps.index(timestamp)
