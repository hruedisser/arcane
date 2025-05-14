from pathlib import Path

from ...data.data_utils.event import Event, EventCatalog
from ..base_sensors.catalog_dataset import CatalogDataset


class ELEVO_EventCatalog(EventCatalog):
    def __init__(
        self,
        folder_path: str | Path = "data/elevo_arrival_times_l1.txt",
        resample_freq: str = "10min",
        event_types: str = "ARRIVAL",
        catalog_name: str = "ELEVO",
        spacecraft: str = "Wind",
        startname: str = "arrival_minus_err",
        endname: str = "arrival_plus_err",
    ):
        super().__init__(
            folder_path=folder_path,
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

        raise NotImplementedError("ELEVO catalog not implemented yet")


class ELEVO_Dataset(CatalogDataset):
    def __init__(
        self,
        folder_path: str | Path = "data/elevo_arrival_times_l1.txt",
        resample_freq: str = "10min",
        event_types: str = "ARRIVAL",
        catalog_name: str = "ELEVO",
        spacecraft: str = "Wind",
        startname: str = "arrival_minus_err",
        endname: str = "arrival_plus_err",
    ):
        self.catalog = ELEVO_EventCatalog(
            folder_path=folder_path,
            event_types=event_types,
            catalog_name=catalog_name,
            spacecraft=spacecraft,
            startname=startname,
            endname=endname,
        )

        super().__init__(
            folder_path=folder_path,
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
