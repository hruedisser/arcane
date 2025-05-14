from pathlib import Path

from ..base_sensors.insitu_dataset import InsituDataset


class RealtimeInsituDataset(InsituDataset):
    def __init__(
        self,
        folder_path: str | Path = "data/noaa_archive_gsm.p",
        components: list[str] = [
            "bx",
            "by",
            "bz",
            "bt",
            "vt",
            "np",
            "tp",
            "beta",
            "pdyn",
            "texrat",
            "source_mag",
            "source_plasma",
        ],
        shift_hours: int = 15,
        resample: bool = True,
        resample_freq: str = "10min",
        resample_method: str = "mean",
        padding: str = "zero",
        lin_interpol: int = 0,
        scaling: str = "Standard",
        scaler_path: str | Path = None,
    ):
        super().__init__(
            folder_path=folder_path,
            components=components,
            shift_hours=shift_hours,
            resample=resample,
            resample_freq=resample_freq,
            resample_method=resample_method,
            padding=padding,
            lin_interpol=lin_interpol,
            scaling=scaling,
            scaler_path=scaler_path,
        )

    @property
    def satellite_name(self) -> str:
        return "NOAA Realtime Archive"
