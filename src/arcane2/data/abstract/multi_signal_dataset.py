from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .dataset_base import DatasetBase


class MultiSignalDataset(DatasetBase):
    """
    Dataset class for handling multiple signal datasets.
    """

    def __init__(
        self,
        single_signal_datasets: List[DatasetBase],
        aggregation: str = "common",
        fill: str = "none",
        time_cut: int = 60,  # in minutes
        save_df: bool = True,
        key_components: List[str] = [],
        key_methods: List[str] = [],
        catalog_idx: List[int] = [],
    ):
        """
        Initializes the MultiSignalDataset.

        Args:
            single_signal_datasets (list): List of DatasetBase objects representing single signal datasets.
            aggregation (str): Aggregation method for timestamps ("all", "common", "I:<idx>").
            fill (str): Method for filling missing timestamps ("none", "last", "closest").
            time_cut (int): Time cut-off in minutes for "closest" fill method.
            save_df (bool): Whether to save the merged data as a DataFrame.
        """

        super().__init__()

        self.single_signal_datasets = single_signal_datasets
        self.aggregation = aggregation
        self.fill = fill
        self.time_cut = time_cut

        # Create a DataFrame to store timestamps and corresponding indices
        logger.info("Initializing timestamps...")
        self._timestamps = self._initialize_timestamps()

        logger.info("Initializing data dictionary...")
        self.data_dict = self._initialize_data_dict()

        logger.info("Creating DataFrame for visualization...")
        self.df = self._initialize_df()

        self.df, self.single_signal_datasets[catalog_idx] = self.single_signal_datasets[
            catalog_idx
        ].add_key_param(
            self.df,
            key_components,
            key_methods,
        )

        self.single_signal_datasets[0].get_data(0)

    def _initialize_timestamps(self) -> list[int]:
        """
        Initializes the DataFrame to store timestamps and corresponding indices.
        """

        if self.aggregation == "all":
            merged_timestamps = self.single_signal_datasets[0].timestamps

            for ds in tqdm(self.single_signal_datasets[1:], desc="Merging timestamps"):
                i, j = 0, 0

                timestmaps_to_merge = ds.timestamps
                tmp_merged_timestamps = [
                    min(merged_timestamps[0], timestmaps_to_merge[0])
                ]

                with tqdm(
                    total=len(merged_timestamps) + len(timestmaps_to_merge)
                ) as pbar:
                    while i < len(merged_timestamps) and j < len(timestmaps_to_merge):

                        if tmp_merged_timestamps[-1] == merged_timestamps[i]:
                            i += 1
                            pbar.update(1)
                            continue

                        if tmp_merged_timestamps[-1] == timestmaps_to_merge[j]:
                            j += 1
                            pbar.update(1)
                            continue

                        if merged_timestamps[i] < timestmaps_to_merge[j]:
                            tmp_merged_timestamps.append(merged_timestamps[i])
                            i += 1
                        else:
                            tmp_merged_timestamps.append(timestmaps_to_merge[j])
                            j += 1
                        pbar.update(1)

                # Append any remaining elements from list1 or list2
                tmp_merged_timestamps.extend(merged_timestamps[i:])
                tmp_merged_timestamps.extend(timestmaps_to_merge[j:])
                merged_timestamps = tmp_merged_timestamps

            return merged_timestamps

        elif self.aggregation == "common":
            # Initialize the merged_timestamps with the timestamps from the first dataset
            merged_timestamps = self.single_signal_datasets[0].timestamps

            # Iterate through the remaining datasets
            for ds in tqdm(self.single_signal_datasets[1:], desc="Merging timestamps"):
                # Find the intersection of merged_timestamps and the current dataset's timestamps
                merged_timestamps = np.intersect1d(merged_timestamps, ds.timestamps)

            merged_timestamps = merged_timestamps.tolist()

            return merged_timestamps

        elif self.aggregation.startswith("I:"):
            # For 'I:<idx>' aggregation, mark timestamps from a specific dataset
            idx = int(self.aggregation[2:])
            return self.single_signal_datasets[idx].timestamps

        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")

    def _initialize_data_dict(self) -> dict[int, list[int | None]]:
        """
        Fills data vs timestamps based on the specified fill method.
        """

        data_dict = {
            t: [None] * len(self.single_signal_datasets) for t in self.timestamps
        }
        for i, ds in tqdm(enumerate(self.single_signal_datasets), desc="Filling:"):
            for j, timestamp in tqdm(enumerate(ds.timestamps), desc=f"Dataset: {i}"):
                if timestamp in data_dict:
                    data_dict[timestamp][i] = j

        if self.fill == "none":
            return data_dict

        elif self.fill == "last":
            min_common_timestamp = max(
                [ds.get_timestamp(0) for ds in self.single_signal_datasets]
            )
            for i in tqdm(range(len(self.timestamps)), desc="Filling:"):
                if self.timestamps[i] < min_common_timestamp:
                    data_dict[self.timestamps[i + 1]] = data_dict[self.timestamps[i]]
                    del data_dict[self.timestamps[i]]
                    del self.timestamps[i]

            for i, ds in tqdm(enumerate(self.single_signal_datasets), desc="Filling:"):
                for j, timestamp in tqdm(enumerate(self.timestamps)):
                    if data_dict[timestamp][i] is None:
                        data_dict[timestamp][i] = data_dict[self.timestamps[j - 1]][i]

            return data_dict

        elif self.fill == "closest":
            for i, ds in tqdm(enumerate(self.single_signal_datasets), desc="Filling:"):
                global_timestamp_idx, ds_timestamp_idx = 0, 1

                while global_timestamp_idx < len(self.timestamps):
                    timestamps_to_fill = self.timestamps[global_timestamp_idx]

                    while (
                        ds.get_timestamp(ds_timestamp_idx - 1) < timestamps_to_fill
                        and ds.get_timestamp(ds_timestamp_idx) < timestamps_to_fill
                        and ds_timestamp_idx < len(ds) - 1
                    ):
                        ds_timestamp_idx += 1

                    if abs(
                        ds.get_timestamp(ds_timestamp_idx - 1) - timestamps_to_fill
                    ) < abs(ds.get_timestamp(ds_timestamp_idx) - timestamps_to_fill):
                        data_dict[timestamps_to_fill][i] = ds_timestamp_idx - 1
                    else:
                        data_dict[timestamps_to_fill][i] = ds_timestamp_idx

                    global_timestamp_idx += 1

            return data_dict

        elif self.fill == "zero":
            # Set all None values to 0
            for timestamp in tqdm(self.timestamps, desc="Filling with zeros"):
                data_dict[timestamp] = [
                    0 if v is None else v for v in data_dict[timestamp]
                ]

            return data_dict

    def _initialize_df(self):
        """
        Creates a dataframe that stores the merged data.
        """

        data = []
        for i in tqdm(range(len(self))):
            data.append(self.get_data(i, scaled=False))

        df = pd.DataFrame(
            data, index=pd.to_datetime(self.timestamps, unit="s").to_pydatetime()
        )
        return df

    @property
    def timestamps(self) -> List[int]:
        """
        Returns the list of unix timestamps.
        """
        return self._timestamps

    def __len__(self) -> int:
        """
        Returns the number of timestamps.
        """
        return len(self.timestamps)

    def get_data(self, idx: int, scaled=True) -> dict:
        """
        Retrieves the data at the specified index.

        Args:
            idx (int): Index of the timestamp.

        Returns:
            dict: Dictionary containing data from all datasets at the specified timestamp.
        """
        data_slice = self.data_dict[self.timestamps[idx]]
        data_dict = {}
        for i, ds in enumerate(self.single_signal_datasets):
            if data_slice[i] is None:
                for k in ds.sensor_ids:
                    data_dict[f"{ds.satellite_name}_{ds.id}-{k}"] = None
            else:
                data = ds.get_data(data_slice[i], scaled)

                for k in data:
                    data_dict[f"{ds.satellite_name}_{ds.id}-{k}"] = data[k]

        return data_dict

    def get_timestamp(self, idx: int) -> int:
        """
        Retrieves the timestamp at the specified index.

        Args:
            idx (int): Index of the timestamp.

        Returns:
            int: Timestamp at the specified index.
        """
        return self.timestamps[idx]

    def get_timestamp_idx(self, timestamp: int) -> int:
        """
        Retrieves the index of the specified timestamp.

        Args:
            timestamp (pd.Timestamp): Timestamp to find the index for.

        Returns:
            int: Index of the specified timestamp.
        """
        return self.timestamps.index(timestamp)

    def __repr__(self) -> str:
        """
        Returns a string representation of the MultiSignalDataset object.
        """
        print_string = f"MultiSignalDataset - {len(self)} samples\nDatasets: {len(self.single_signal_datasets)}\n"
        for i, d in enumerate(self.single_signal_datasets):
            inner_repr = repr(d)
            lines = inner_repr.split("\n")
            inner_repr = "\n".join(["\t" + line for line in lines])

            print_string += f"\n{i} -------------\n"
            print_string += inner_repr
        print_string += "\n------------------\n"
        return print_string

    @property
    def id(self):
        """
        Returns the ID of the dataset.
        """
        return ""
