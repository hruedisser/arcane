from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..abstract.dataset_base import DatasetBase
from ..data_utils.event import EventCatalog


class CatalogDataset(DatasetBase):
    def __init__(
        self,
        folder_paths: List[str | Path],
        resample_freq: str = "10min",
        event_types: str = "CME",
        catalog_name: str = "ICMECAT",
        spacecraft: str = "Wind",
        startname: str = "icme_start_time",
        endname: str = "mo_end_time",
        catalog: EventCatalog = [],
        filters: bool = False,
        cap: float = 0.5,
        stats_dict: dict = {},
    ):
        super().__init__()

        try:
            self.folder_paths = [Path(folder_path) for folder_path in folder_paths]
        except TypeError:
            self.folder_paths = folder_paths

        self.resample_freq = resample_freq
        self.event_types = event_types
        self.catalog_name = catalog_name
        self.spacecraft = spacecraft
        self.startname = startname
        self.endname = endname
        self.catalog = catalog

        self.filters = filters
        self.cap = cap

        logger.info(f"Creating dataset from {self.catalog}")
        self.data = self.create_data(filters=filters, cap=cap)

        self.data["timestamp"] = self.data.index.astype("int64") // 10**9

        self.data.reset_index(drop=True, inplace=True)

        self._timestamps = self.data["timestamp"].values

    def __len__(self) -> int:
        return len(self.data)

    @property
    def timestamps(self):
        return self._timestamps

    def get_data(self, idx, scaled=True):
        data = {
            f"{component}": self.data[component].iloc[idx]
            for component in self.data.columns
            if "time" not in component
        }
        return data

    @property
    def sensor_ids(self):
        return {self.event_types, self.spacecraft}

    def get_timestamp(self, idx):
        return self.data["timestamp"].values[idx]

    def get_timestamp_idx(self, timestamp):
        return self.data[self.data["timestamp"] == timestamp].index[0]

    def create_data(self, filters=False, cap=0.5) -> pd.DataFrame:
        # Create a DatetimeIndex with the specified date range
        date_range = pd.date_range(
            start=self.catalog.begin.round(self.resample_freq),
            end=self.catalog.end.round(self.resample_freq),
            freq=self.resample_freq,
        )

        # Create an empty DataFrame with the specified DatetimeIndex
        data = pd.DataFrame(
            index=date_range, columns=[self.catalog.event_types, "event_id"]
        ).tz_localize(None)
        data[self.catalog.event_types] = 0

        # Set all "event_id" values to ""
        data["event_id"] = ""

        # Iterate over each event in the catalog
        for event in tqdm(self.catalog.event_cat):
            # Create a mask for the event's duration
            mask = (data.index >= event.begin.round(self.resample_freq)) & (
                data.index <= event.end.round(self.resample_freq)
            )

            # Set the event's type in the DataFrame
            data.loc[mask, event.event_type] = 1
            data.loc[mask, "event_id"] = event.event_id

            if filters:
                second_part_start = event.begin.round(
                    self.resample_freq
                ) + event.duration * (1 - cap)

                mask = (data.index >= second_part_start) & (
                    data.index <= event.end.round(self.resample_freq)
                )

                data.loc[mask, event.event_type] = 2

        return data

    @property
    def id(self):
        return "catalog"

    def add_key_param(
        self, df: pd.DataFrame, components: List[str], methods: List[str] = "min"
    ):
        """
        Add a key parameter to the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            component (str): Component to calculate the key parameter for.
            method (str): Method to calculate the key parameter. Defaults to "min".

        Returns:
            pd.DataFrame: DataFrame with the key parameter added.
        """
        new_components = []
        new_keys = []

        for i, component in enumerate(components):
            new_component = [col for col in df.columns if component in col][0]
            new_components.append(new_component)
            new_key = f"key_param-{component}-{methods[i]}"
            new_keys.append(new_key)

        logger.info(f"Adding key parameters {new_keys} to the dataset")

        for key in new_keys:
            self.data[key] = 0.0
            df[key] = 0.0

        self.data.index = self.data["timestamp"]

        for event in tqdm(self.catalog.event_cat):
            event_start = event.begin.timestamp()
            event_end = event.end.timestamp()

            # Create a mask for the event's duration
            mask = (self.data["timestamp"] > event_start) & (
                self.data["timestamp"] < event_end
            )

            subdf = df[(df.index >= event.begin) & (df.index <= event.end)]
            if not subdf.empty:
                for i, component in enumerate(new_components):
                    key_value = getattr(subdf[component], methods[i])()
                    if np.isnan(key_value):
                        key_value = 0.0
                    self.data.loc[mask, new_keys[i]] = key_value

                    df.loc[
                        (df.index >= event.begin) & (df.index <= event.end), new_keys[i]
                    ] = key_value
        self.data.reset_index(drop=True, inplace=True)

        return df, self
