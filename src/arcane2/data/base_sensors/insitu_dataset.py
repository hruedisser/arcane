import pickle
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..abstract.dataset_base import DatasetBase
from ..data_utils import features


class InsituDataset(DatasetBase):
    def __init__(
        self,
        folder_path: str | Path,
        components: list[str] = ["bx", "by", "bz", "bt"],
        shift_hours: int = 15,
        resample: bool = True,
        resample_freq: str = "10min",
        resample_method: str = "mean",
        padding: str = "drop",
        lin_interpol: int = 0,
        scaling: str = "Standard",
        scaler_path: str | Path = None,
    ):
        """
        Initializes insitu dataset.

        Args:
            folder_path (str | Path): Path to folder containing insitu data.
            components (list[str], optional): Components of insitu data. Defaults to [Bx, By, Bz, Bt].
        """
        super().__init__()

        self.folder_path = folder_path
        self.components = components
        self.shift_hours = shift_hours
        self.resample = resample
        self.resample_freq = resample_freq
        self.resample_method = resample_method
        self.padding = padding
        self.lin_interpol = lin_interpol
        self.scaling = scaling
        self.scaler_path = Path(scaler_path) if scaler_path else None

        self.data = self.load_data()
        # Convert DatetimeIndex to integer timestamps
        self.data["timestamp"] = self.data.index.astype("int64") // 10**9

        self.data.reset_index(drop=True, inplace=True)
        self.unscaled_data.reset_index(drop=True, inplace=True)

        self._timestamps = self.data["timestamp"].values

    @property
    def timestamps(self):
        return self._timestamps

    def load_data(self) -> pd.DataFrame:
        """
        Load insitu data from folder_path.

        Returns:
            pd.DataFrame: Insitu data.
        """
        logger.info(f"Loading insitu data from {self.folder_path}")

        try:
            if self.folder_path.startswith("http"):
                file = urllib.request.urlopen(self.folder_path)
                data, dh = pickle.load(file)
            else:
                self.folder_path = Path(self.folder_path)
                try:
                    [data, _] = pickle.load(open(self.folder_path, "rb"))
                except ValueError:
                    data = pickle.load(open(self.folder_path, "rb"))
        except Exception:
            self.folder_path = Path(self.folder_path)
            try:
                [data, _] = pickle.load(open(self.folder_path, "rb"))
            except ValueError:
                data = pickle.load(open(self.folder_path, "rb"))

        dataframe = pd.DataFrame(data)
        dataframe.set_index("time", inplace=True)
        dataframe.index.name = None
        dataframe.index = dataframe.index.tz_localize(None)

        # Resample the dataframe if required
        if self.resample:
            logger.info(
                f"Resampling data with frequency '{self.resample_freq}' using method '{self.resample_method}'"
            )
            dataframe = dataframe.resample(self.resample_freq).apply(
                self.resample_method
            )
        else:
            # Resample to the smallest sampling present in the dataset
            smallest_sampling = dataframe.index.to_series().diff().min()
            dataframe = dataframe.resample(smallest_sampling).mean()
            logger.info(f"Data resampled to: {smallest_sampling}")

        if "beta" in self.components:
            logger.info("Computing beta")
            dataframe = features.computeBetawiki(dataframe)
        if "pdyn" in self.components:
            logger.info("Computing pdyn")
            dataframe = features.computePdyn(dataframe)
        if "texrat" in self.components:
            logger.info("Computing texrat")
            dataframe = features.computeTexrat(dataframe)
        if "bz_q25" in self.components:
            logger.info("Computing quartiles")
            dataframe = features.computeQuartiles(
                dataframe, shifthours=self.shift_hours
            )
        if "bz_negcumsum" in self.components:
            logger.info("Computing negative cumsum")
            dataframe = features.computeCumsumMax(dataframe)

        possible_key_components = [
            "bz_min",
            "bz_mean",
            "bt_max",
            "bt_mean",
            "bz_std",
            "bt_std",
        ]

        key_components = [
            component
            for component in possible_key_components
            if component in self.components
        ]

        if len(key_components) > 0:
            logger.info("Computing key params")
            dataframe = features.computekeyparams(dataframe, key_components)

        # Check if all components in self.components are in the dataframe columns
        missing_components = set(self.components) - set(dataframe.columns)
        if missing_components:
            raise ValueError(f"Missing components: {', '.join(missing_components)}")

        dataframe = dataframe[self.components]

        if self.lin_interpol > 0:
            logger.info(f"Interpolating missing values with limit {self.lin_interpol}")
            dataframe = dataframe.interpolate(method="time", limit=self.lin_interpol)

        # Handle padding if required
        if self.padding:
            padding_methods = {
                "drop": dataframe.dropna,
                "ffill": dataframe.ffill,
                "bfill": dataframe.bfill,
                "both": lambda: dataframe.ffill().bfill(),
                "zero": lambda: dataframe.fillna(0),
            }
            padding_func = padding_methods.get(self.padding)
            if padding_func:
                logger.info(f"Padding data using method '{self.padding}'")
                dataframe = padding_func()  # Call the padding function
            else:
                logger.warning(f"Unknown padding method: {self.padding}")

        # Save unscaled data before applying scaling
        self.unscaled_data = dataframe.copy()

        vector_components = dataframe[self.components[:4]].values

        new_dataframe = pd.DataFrame(
            index=dataframe.index,
            columns=self.components,
        )

        # Load or fit the scaler and apply scaling
        if self.scaling == "Standard":
            mean = np.nanmean(vector_components.flatten())
            std = np.nanstd(vector_components.flatten())
            scaled_vector_components = (vector_components - mean) / std

            new_dataframe[self.components[:4]] = scaled_vector_components
            if "bz_q25" in self.components:
                for column in self.components[4:-6]:
                    new_dataframe[column] = (
                        dataframe[column] - np.nanmean(dataframe[column].values)
                    ) / np.nanstd(dataframe[column].values)
                for column in self.components[-6:]:
                    new_dataframe[column] = dataframe[column]
            elif "bz_negcumsum" in self.components:
                for column in self.components[4:-2]:
                    new_dataframe[column] = (
                        dataframe[column] - np.nanmean(dataframe[column].values)
                    ) / np.nanstd(dataframe[column].values)
                for column in self.components[-2:]:
                    new_dataframe[column] = dataframe[column]
            elif len(key_components) > 0:
                for column in self.components[4 : -len(key_components)]:
                    new_dataframe[column] = (
                        dataframe[column] - np.nanmean(dataframe[column].values)
                    ) / np.nanstd(dataframe[column].values)
                for column in self.components[-len(key_components) :]:
                    new_dataframe[column] = dataframe[column]
            else:
                for column in self.components[4:]:
                    new_dataframe[column] = (
                        dataframe[column] - np.nanmean(dataframe[column].values)
                    ) / np.nanstd(dataframe[column].values)

        elif self.scaling == "None":
            for column in self.components:
                new_dataframe[column] = dataframe[column]

        if new_dataframe.isnull().values.any():
            logger.warning("Dataframe still contains NaN values")

        return new_dataframe

    def __len__(self):
        return len(self.data)

    def get_data(self, idx, scaled=True):
        """
        Retrieve data at a given index.

        Args:
            idx (int): Index of the data.
            scaled (bool): Whether to return scaled or unscaled data. Defaults to True.

        Returns:
            dict: Data for the specified components.
        """
        data_source = self.data if scaled else self.unscaled_data
        data = {
            f"{component}": data_source[component].iloc[idx]
            for component in self.components
        }

        return data

    @property
    def sensor_ids(self):
        return self.components

    def get_timestamp(self, idx):
        return self.data["timestamp"].values[idx]

    def get_timestamp_idx(self, timestamp):
        return self.data[self.data["timestamp"] == timestamp].index[0]

    @property
    def timestamps(self):
        return self.data["timestamp"].values.tolist()

    @property
    def id(self):
        return "insitu"
