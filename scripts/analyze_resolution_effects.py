import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .arcane2.data.data_utils.event import Event

insitu_data_path = Path("data/noaa_archive_gsm.p")

[data, _] = pickle.load(open(insitu_data_path, "rb"))

dataframe = pd.DataFrame(data)
dataframe.set_index("time", inplace=True)
dataframe.index.name = None
dataframe.index = dataframe.index.tz_localize(None)


icme_catalog_data_path = Path("data/dataverse_files/ICME_catalog_OMNI.csv")
sheath_catalog_data_path = Path("data/dataverse_files/Sheath_catalog_OMNI.csv")

icme_df = pd.read_csv(icme_catalog_data_path, index_col=0, header=0, sep=";")
icme_df[icme_df.columns[0]] = pd.to_datetime(
    icme_df[icme_df.columns[0]], format="%d/%m/%Y %H:%M"
)
icme_df[icme_df.columns[1]] = pd.to_datetime(
    icme_df[icme_df.columns[1]], format="%d/%m/%Y %H:%M"
)

sheath_df = pd.read_csv(sheath_catalog_data_path, index_col=0, header=0, sep=";")
sheath_df[sheath_df.columns[0]] = pd.to_datetime(
    sheath_df[sheath_df.columns[0]], format="%d/%m/%Y %H:%M"
)
sheath_df[sheath_df.columns[1]] = pd.to_datetime(
    sheath_df[sheath_df.columns[1]], format="%d/%m/%Y %H:%M"
)

icme_evtlist = [
    Event(
        icme_df[icme_df.columns[0]][i],
        icme_df[icme_df.columns[1]][i],
        "CME",
        "Wind",
        "NGUYEN",
        f"NGUYEN_CME_{i}",
    )
    for i in range(0, len(icme_df))
]

sheath_evtlist = [
    Event(
        sheath_df[sheath_df.columns[0]][i],
        sheath_df[sheath_df.columns[1]][i],
        "Sheath",
        "Wind",
        "NGUYEN",
        f"NGUYEN_Sheath_{i}",
    )
    for i in range(0, len(sheath_df))
]

events_with_sheath = []
events_without_sheath = []

for icme_event in icme_evtlist:
    # Check if there is a sheath that ends with the beginning of the ICME event
    sheath_candidates = [
        sheath_event
        for sheath_event in sheath_evtlist
        if sheath_event.end == icme_event.begin
    ]

    if (
        icme_event.begin > dataframe.index.min()
        and icme_event.end < dataframe.index.max()
    ):
        if sheath_candidates:
            # If there is a sheath, add the ICME event with the sheath
            events_with_sheath.append((sheath_candidates[0], icme_event))
        else:
            # If there is no sheath, add the ICME event without a sheath
            events_without_sheath.append(icme_event)


plot_path = Path("plots/events")
plot_path.mkdir(parents=True, exist_ok=True)


max_plots = 1000
plot_delta = 6

resolutions = [1, 10, 30]  # in minutes
variables = {
    0: {
        "label": "B [nT]",
        "components": ["bx", "by", "bz", "bt"],
        "colors": ["red", "green", "blue", "black"],
    },
    1: {"label": "V [km/s]", "components": ["vt"], "colors": ["black"]},
    2: {"label": "N$_{P}$ [cm$^{-3}$]", "components": ["np"], "colors": ["black"]},
    3: {"label": "T$_{P}$ [K]", "components": ["tp"], "colors": ["black"]},
}

for i, (sheath_event, icme_event) in enumerate(events_with_sheath):
    if i >= max_plots:
        break

    print(f"Sheath Event {i+1}: {sheath_event}")
    print(f"ICME Event {i+1}: {icme_event}")

    plot_data = dataframe[
        sheath_event.begin
        - datetime.timedelta(hours=plot_delta) : icme_event.end
        + datetime.timedelta(hours=plot_delta)
    ]

    fig, axes = plt.subplots(
        len(variables), len(resolutions), figsize=(15, 10), sharex=True
    )

    for j, resolution in enumerate(resolutions):
        resampled_data = plot_data.resample(f"{resolution}T").mean()

        for k, var in variables.items():

            var_cols = var["components"]
            var_name = var["label"]
            colors = var["colors"]

            ax = axes[k, j]
            for col, color in zip(var_cols, colors):
                ax.plot(
                    resampled_data.index, resampled_data[col], label=col, color=color
                )

            if k == 0:
                ax.set_title(f"Resolution: {resolution} min")

            ax.set_ylabel(var_name)

            ax.axvline(x=sheath_event.begin, color="red", linestyle="--")
            ax.axvline(x=icme_event.begin, color="black", linestyle="--")
            ax.axvline(x=icme_event.end, color="black", linestyle="--")
            ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path / f"event_{i+1}_resolution_effects.png")
    plt.close(fig)

print("Plots saved in:", plot_path.resolve())
print("Analysis complete.")
