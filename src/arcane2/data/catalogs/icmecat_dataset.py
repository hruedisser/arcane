from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ...data.data_utils.event import Event, EventCatalog
from ..base_sensors.catalog_dataset import CatalogDataset

file_dir = Path(__file__).resolve()
data_dir = file_dir.parents[4] / "data"


class ICMECAT_EventCatalog(EventCatalog):
    def __init__(
        self,
        folder_paths: str | Path = "data/HELIO4CAST_ICMECAT_v23.csv",
        event_types: str = "CME",
        catalog_name: str = "ICMECAT",
        spacecraft: str = "Wind",
        startname: str = "icme_start_time",
        endname: str = "mo_end_time",
    ):
        super().__init__(
            folder_paths=folder_paths,
            event_types=event_types,
            catalog_name=catalog_name,
            spacecraft=spacecraft,
            startname=startname,
            endname=endname,
        )

    def read_catalog(self) -> list[Event]:
        """
        Read catalog from folder_path
        """
        evtlist = []

        # load from local file if url not reachable
        try:
            ic = pd.read_csv(self.folder_paths[0])
        except:
            print(f"Could not reach {self.folder_paths[0]}, loading local file...")
            file_name = self.folder_paths[0].split("/")[-1]
            ic = pd.read_csv(data_dir / file_name)

        isc = ic.loc[:, "sc_insitu"]
        iid = ic.loc[:, "icmecat_id"]
        begin = pd.to_datetime(ic.loc[:, self.startname])
        end = pd.to_datetime(ic.loc[:, self.endname])

        iind = np.where(isc == self.spacecraft)[0]
        for i in tqdm(iind):
            evtlist.append(
                Event(
                    begin[i],
                    end[i],
                    self.event_types,
                    self.spacecraft,
                    self.catalog_name,
                    iid[i],
                )
            )

        logger.info(f"Sorting events by {self.startname}")
        evtlist.sort(key=lambda event: event.begin)  # Sort events by begin

        return evtlist


class ICMECAT_Dataset(CatalogDataset):
    def __init__(
        self,
        folder_paths: str | Path = "data/HELIO4CAST_ICMECAT_v22.csv",
        resample_freq: str = "10min",
        event_types: str = "CME",
        catalog_name: str = "ICMECAT",
        spacecraft: str = "Wind",
        startname: str = "icme_start_time",
        endname: str = "mo_end_time",
    ):
        self.catalog = ICMECAT_EventCatalog(
            folder_paths=folder_paths,
            event_types=event_types,
            catalog_name=catalog_name,
            spacecraft=spacecraft,
            startname=startname,
            endname=endname,
        )

        super().__init__(
            folder_paths=[folder_paths],
            resample_freq=resample_freq,
            event_types=event_types,
            catalog_name=catalog_name,
            spacecraft=spacecraft,
            startname=startname,
            endname=endname,
            catalog=self.catalog,
        )

    @property
    def satellite_name(self) -> str:
        return f"{self.catalog_name}"
