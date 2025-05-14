from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from ...data.data_utils.event import Event, EventCatalog, merge_sheath
from ..base_sensors.catalog_dataset import CatalogDataset


class Nguyen_EventCatalog(EventCatalog):
    def __init__(
        self,
        folder_paths: List[str | Path] = ["data/dataverse_files/ICME_catalog_OMNI.csv"],
        event_types: str = "CME",
        catalog_name: str = "NGUYEN",
        spacecraft: str = "Wind",
        startname: str = "begin",
        endname: str = "end",
    ):
        super().__init__(
            folder_paths=folder_paths,
            event_types=event_types,
            catalog_name=catalog_name,
            spacecraft=spacecraft,
            startname=startname,
            endname=endname,
        )

    def read_catalog(
        self,
        index_col=0,
        header=0,
        dateFormat="%d/%m/%Y %H:%M",
        sep=";",
        col_begin=0,
        col_end=1,
        col_proba=None,
    ) -> list[Event]:
        """
        Read catalog from folder_path
        """
        evtlists = []
        for folder_path in self.folder_paths:
            logger.info(f"Reading catalog from {folder_path}")
            df = pd.read_csv(folder_path, index_col=index_col, header=header, sep=sep)
            df[df.columns[col_begin]] = pd.to_datetime(
                df[df.columns[col_begin]], format=dateFormat
            )
            df[df.columns[col_end]] = pd.to_datetime(
                df[df.columns[col_end]], format=dateFormat
            )
            if col_proba:
                evtlist = [
                    Event(
                        df[df.columns[col_begin]][i],
                        df[df.columns[col_end]][i],
                        self.event_types,
                        self.spacecraft,
                        self.catalog_name,
                        f"{self.catalog_name}_{self.event_types}_{i}",
                        df[df.columns[col_proba]][i],
                    )
                    for i in range(0, len(df))
                ]
            else:
                evtlist = [
                    Event(
                        df[df.columns[col_begin]][i],
                        df[df.columns[col_end]][i],
                        self.event_types,
                        self.spacecraft,
                        self.catalog_name,
                        f"{self.catalog_name}_{self.event_types}_{i}",
                    )
                    for i in range(0, len(df))
                ]
            logger.info(f"Sorting events by {self.startname}")
            evtlist.sort(key=lambda event: event.begin)  # Sort events by begin
            evtlists.append(evtlist)

        evtlist = evtlists[0]
        if len(evtlists) > 1:
            for i, elt in enumerate(evtlist):
                for k in range(len(evtlists[1])):
                    if evtlists[1][k].end == elt.begin:
                        evtlist[i] = merge_sheath(evtlist[i], evtlists[1][k])
        return evtlist


class Nguyen_Dataset(CatalogDataset):
    def __init__(
        self,
        folder_paths: List[str | Path] = ["data/HELIO4CAST_ICMECAT_v22.csv"],
        resample_freq: str = "10min",
        event_types: str = "CME",
        catalog_name: str = "NGUYEN",
        spacecraft: str = "Wind",
        startname: str = "begin",
        endname: str = "end",
        filters: bool = False,
        cap: float = 0.5,
    ):
        self.catalog = Nguyen_EventCatalog(
            folder_paths=folder_paths,
            event_types=event_types,
            catalog_name=catalog_name,
            spacecraft=spacecraft,
            startname=startname,
            endname=endname,
        )

        super().__init__(
            folder_paths=folder_paths,
            resample_freq=resample_freq,
            event_types=event_types,
            catalog_name=catalog_name,
            spacecraft=spacecraft,
            startname=startname,
            endname=endname,
            catalog=self.catalog,
            filters=filters,
            cap=cap,
        )

    @property
    def satellite_name(self) -> str:
        return f"{self.catalog_name}"
