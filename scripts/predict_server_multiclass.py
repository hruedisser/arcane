import datetime
import os
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from plotly.subplots import make_subplots
from pytorch_lightning import seed_everything
from scipy.constants import k, pi, proton_mass
from tqdm import tqdm

from .arcane2.data.abstract.multi_signal_dataset import MultiSignalDataset
from .arcane2.data.abstract.sequential_dataset import SequentialDataset
from .arcane2.data.catalogs.icmecat_dataset import ICMECAT_Dataset
from .arcane2.data.data_utils.event import EventCatalog
from .arcane2.data.datamodule import ParsedDataModule
from .arcane2.data.realtime.realtime_insitu_dataset import RealtimeInsituDataset
from .arcane2.data.utils import get_best_model_path

train = False


def create_group_boundaries(years_group):
    if not years_group:
        return []
    boundaries = []
    start_year = years_group[0]
    end_year = years_group[0]
    for year in years_group[1:]:
        if year - end_year == 1:
            end_year = year
        else:
            boundaries.append([f"{start_year}0101T000000", f"{end_year+1}0101T000000"])
            start_year = year
            end_year = year
    boundaries.append([f"{start_year}0101T000000", f"{end_year+1}0101T000000"])
    return boundaries


def update_event_catalog(file, catalog):
    """
    Update the event catalog with the new events.
    """

    evtlist = []
    format = "%Y-%m-%dT%H:%MZ"

    new_evtlist = [event for event in catalog]

    catalog_data = pd.DataFrame(
        data=[
            [
                datetime.datetime.strftime(event.begin, format),
                datetime.datetime.strftime(event.end, format),
                event.event_type,
                event.spacecraft,
                event.catalog,
                event.event_id,
                event.proba,
                event.proba_max,
            ]
            for event in new_evtlist
        ],
        columns=[
            "ICME_start_time",
            "ICME_end_time",
            "event_type",
            "spacecraft",
            "catalog",
            "event_id",
            "mean_probability",
            "max_probability",
        ],
        index=range(len(new_evtlist)),
    )

    # sort events by begin time with the most recent event first
    catalog_data = catalog_data.sort_values(by="ICME_start_time", ascending=False)

    catalog_data.to_csv(file, index=False)


@hydra.main(config_path="../config")
def main(cfg):
    """
    Main function for training the model.
    """

    seed_everything(42, workers=True)

    #####################################
    #####################################
    ############ PREDICTION #############
    #####################################
    #####################################

    device = "cpu"

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        prog_bar = True  # False if torch.cuda.is_available() else True

    else:
        device = torch.device(device)
        prog_bar = True

    print(f"Using device: {device}")

    # Access the overrides from HydraConfig
    hydra_cfg = HydraConfig.get()

    cache_path = (
        Path(hydra.utils.get_original_cwd() + cfg["cache_folder"]) / cfg["run_name"]
    )

    url = "https://helioforecast.space/static/sync/insitu_python/noaa_rtsw_last_35files_now.p"

    insitu_cfg = cfg["base_dataset"]["dataset"]["single_signal_datasets"][2]

    insitu_data = RealtimeInsituDataset(
        folder_path=url,
        components=insitu_cfg["components"],
        resample=insitu_cfg["resample"],
        resample_freq=insitu_cfg["resample_freq"],
        resample_method=insitu_cfg["resample_method"],
        padding=insitu_cfg["padding"],
        lin_interpol=insitu_cfg["lin_interpol"],
    )

    high_res_insitu_data = RealtimeInsituDataset(
        folder_path=url,
        components=insitu_cfg["components"],
        resample=False,
        padding=insitu_cfg["padding"],
        lin_interpol=insitu_cfg["lin_interpol"],
    )

    catalog_cfg_0 = cfg["base_dataset"]["dataset"]["single_signal_datasets"][0]
    catalog_cfg_1 = cfg["base_dataset"]["dataset"]["single_signal_datasets"][1]

    catalog_dataset_0 = ICMECAT_Dataset(
        folder_paths=catalog_cfg_0["folder_paths"],
        event_types=catalog_cfg_0["event_types"],
        startname=catalog_cfg_0["startname"],
        endname=catalog_cfg_0["endname"],
    )

    catalog_dataset_1 = ICMECAT_Dataset(
        folder_paths=catalog_cfg_1["folder_paths"],
        event_types=catalog_cfg_1["event_types"],
        startname=catalog_cfg_1["startname"],
        endname=catalog_cfg_1["endname"],
    )

    multi_signal_dataset = MultiSignalDataset(
        single_signal_datasets=[catalog_dataset_0, catalog_dataset_1, insitu_data],
        catalog_idx=0,
        aggregation="I:2",
        fill="zero",
    )

    multi_signal_dataset_high_res = MultiSignalDataset(
        single_signal_datasets=[
            catalog_dataset_0,
            catalog_dataset_1,
            high_res_insitu_data,
        ],
        catalog_idx=0,
        aggregation="I:2",
        fill="zero",
    )

    sequential_dataset = SequentialDataset(
        multi_signal_dataset,
        n_samples=cfg["base_dataset"]["n_samples"],
        max_time_gap=cfg["base_dataset"]["max_time_gap"],
        filters=False,
        weights=False,
    )

    data_module = ParsedDataModule(
        train_dataset=sequential_dataset,
        val_dataset=sequential_dataset,
        test_dataset=sequential_dataset,
        batch_size=cfg["data_module"]["batch_size"],
        num_workers=cfg["data_module"]["num_workers"],
        shuffle=False,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
        train_collate_fn=instantiate(cfg["data_module"]["train_collate_fn"]),
        val_collate_fn=instantiate(cfg["data_module"]["val_collate_fn"]),
        test_collate_fn=instantiate(cfg["data_module"]["test_collate_fn"]),
    )

    for fold in range(3):

        print(f"Predicting fold {fold}")

        modelname = f"{cfg['run_name']}_{fold}"

        checkpoint_path = cache_path

        modelpath = get_best_model_path(checkpoint_path, modelname)

        module = instantiate(cfg["module"])

        try:
            checkpoint = torch.load(modelpath, map_location=device, weights_only=False)
            module.load_state_dict(checkpoint["state_dict"])
        except:
            model = instantiate(cfg["model"])
            model = torch.load(modelpath, map_location=device, weights_only=False)
            module = instantiate(cfg["module"], model=model)

        logger = None

        # Find the exact override for boundaries in the command line
        for override in hydra_cfg.overrides.task:
            if override.startswith("+base_dataset="):
                diff_name = override.split("=")[1]

                break

        # Set the model to evaluation mode
        module.eval()

        with torch.no_grad():
            timestamps = []

            for i, batch in tqdm(enumerate(data_module.test_dataloader())):
                timestamps.extend(batch["timestamp"].tolist())

            all_results_df_sheath = pd.DataFrame(
                index=pd.to_datetime(timestamps, unit="s")
            )
            all_results_df_mo = pd.DataFrame(index=pd.to_datetime(timestamps, unit="s"))

            for i, batch in tqdm(enumerate(data_module.test_dataloader())):
                insitu_data_batch = batch["insitu"]
                segmentation = batch["catalog"].float().squeeze()
                timestamp = batch["timestamp"]
                idxs = batch["idx"]

                len_insitu = len(segmentation[0])

                seg_hat = module(insitu_data_batch.to(module.device))

                for i, batchitem in enumerate(idxs):

                    # convert idx to int
                    batchitem_int = int(batchitem)

                    # determine start index of the segment
                    start_index = np.max([0, batchitem_int - len_insitu + 1])

                    end_index = batchitem_int + 1

                    considered_index = timestamps[start_index:end_index]

                    num_considered_elements = len(considered_index)

                    considered_seg_sheath = (
                        seg_hat[i, 1, :].squeeze().tolist()[-num_considered_elements:]
                    )
                    considered_seg_mo = (
                        seg_hat[i, 2, :].squeeze().tolist()[-num_considered_elements:]
                    )

                    try:
                        small_df_sheath = pd.DataFrame(
                            considered_seg_sheath,
                            index=pd.to_datetime(considered_index, unit="s"),
                            columns=[f"predicted_value_sheath_{batchitem_int}"],
                        )

                        small_df_mo = pd.DataFrame(
                            considered_seg_mo,
                            index=pd.to_datetime(considered_index, unit="s"),
                            columns=[f"predicted_value_mo_{batchitem_int}"],
                        )

                    except:
                        breakpoint()

                    all_results_df_sheath = pd.concat(
                        [all_results_df_sheath, small_df_sheath], axis=1
                    )
                    all_results_df_mo = pd.concat(
                        [all_results_df_mo, small_df_mo], axis=1
                    )

            all_results_df_sheath.index = all_results_df_sheath.index.round("30min")
            all_results_df_mo.index = all_results_df_mo.index.round("30min")

            aggregated_results_sheath = all_results_df_sheath.mean(axis=1)
            aggregated_results_mo = all_results_df_mo.mean(axis=1)

        if fold == 0:
            df = data_module.test_dataset.dataset.df
        else:
            df = merged_df

        expected_index = pd.date_range(
            start=df.index[0], end=df.index[-1], freq="30min"
        )

        # reindex the dataframe to the expected index
        aggregated_results_sheath = aggregated_results_sheath.reindex(expected_index)
        aggregated_results_mo = aggregated_results_mo.reindex(expected_index)

        df = df.reindex(expected_index, method="ffill")
        aggregated_results_sheath = aggregated_results_sheath.ffill()
        aggregated_results_mo = aggregated_results_mo.ffill()

        # Convert aggregated_results (Series) to a DataFrame
        aggregated_results_df_sheath = aggregated_results_sheath.to_frame(
            name=f"predicted_value_sheath_{fold}"
        )
        aggregated_results_df_mo = aggregated_results_mo.to_frame(
            name=f"predicted_value_mo_{fold}"
        )

        merged_df_sheath = pd.merge(
            aggregated_results_df_sheath,
            df,
            how="outer",
            left_index=True,
            right_index=True,
        )
        merged_df = pd.merge(
            aggregated_results_df_mo,
            merged_df_sheath,
            how="outer",
            left_index=True,
            right_index=True,
        )

    merged_df = merged_df.dropna()

    merged_df["event_id"] = ""

    merged_df["proba"] = ""

    cols_to_merge = [f"predicted_value_sheath_{fold}" for fold in range(3)]
    merged_df["predicted_value_sheath"] = merged_df[cols_to_merge].mean(axis=1)
    merged_df = merged_df.drop(columns=cols_to_merge)

    cols_to_merge = [f"predicted_value_mo_{fold}" for fold in range(3)]
    merged_df["predicted_value_mo"] = merged_df[cols_to_merge].mean(axis=1)
    merged_df = merged_df.drop(columns=cols_to_merge)

    # set merged_df["predicted_value"] and merged_df["predicted_value_threshold"] to nan if the value is smaller than 0.1

    merged_df.loc[
        (merged_df["predicted_value_sheath"] < 0.1),
        "predicted_value_sheath",
    ] = np.nan

    merged_df.loc[
        (merged_df["predicted_value_mo"] < 0.1),
        "predicted_value_mo",
    ] = np.nan

    ###########################
    ###### ADD THRESHOLD ######
    ###########################

    v_threshold = 30 * 1e3

    T_threshold = v_threshold**2 * proton_mass * pi / (8 * k)

    T_threshold = np.round(T_threshold, -3)

    b_threshold = 8
    beta_threshold = 0.3
    v_threshold = 30

    print(
        f"Thresholds: T = {T_threshold} K, B = {b_threshold} nT, beta = {beta_threshold}, V = {v_threshold} km/s"
    )

    merged_df["predicted_value_threshold"] = np.nan

    merged_df.loc[
        (merged_df["NOAA Realtime Archive_insitu-bt"] >= b_threshold)
        & (merged_df["NOAA Realtime Archive_insitu-beta"] <= beta_threshold)
        & (merged_df["NOAA Realtime Archive_insitu-tp"] <= T_threshold),
        "predicted_value_threshold",
    ] = 1

    ################################
    ###### EXTRACTING CATALOG ######
    ################################

    catalog_sheath = EventCatalog(
        event_types="SHEATH",
        catalog_name="ARCANE",
        spacecraft="RTSW",
        dataframe=merged_df,
        key="predicted_value_sheath",
        creep_delta=30,
        thresh=0.5,
    ).event_cat

    for event in catalog_sheath:
        event.proba = merged_df.loc[
            event.begin : event.end, "predicted_value_sheath"
        ].mean()
        event.proba_max = merged_df.loc[
            event.begin : event.end, "predicted_value_sheath"
        ].max()

    eventcounts = defaultdict(int)

    for event in catalog_sheath:
        event_date = event.begin.date()
        event_date_str = event_date.strftime("%Y%m%d")
        event_number = eventcounts[event_date] + 1
        eventcounts[event_date] = event_number

        event.event_id = f"SHEATH_RTSW_ARCANE_{event_date_str}_{event_number:02d}"

    for timestamp in merged_df.index:
        for event in catalog_sheath:
            if event.begin <= timestamp <= event.end:
                merged_df.loc[timestamp, "event_id"] = event.event_id
                merged_df.loc[timestamp, "proba"] = (
                    "Probability: " + str(event.proba)[:4]
                )

    catalog_mo = EventCatalog(
        event_types="MO",
        catalog_name="ARCANE",
        spacecraft="RTSW",
        dataframe=merged_df,
        key="predicted_value_mo",
        creep_delta=30,
        thresh=0.5,
    ).event_cat

    for event in catalog_mo:
        event.proba = merged_df.loc[
            event.begin : event.end, "predicted_value_mo"
        ].mean()
        event.proba_max = merged_df.loc[
            event.begin : event.end, "predicted_value_mo"
        ].max()

    eventcounts = defaultdict(int)

    for event in catalog_mo:
        event_date = event.begin.date()
        event_date_str = event_date.strftime("%Y%m%d")
        event_number = eventcounts[event_date] + 1
        eventcounts[event_date] = event_number

        event.event_id = f"MO_RTSW_ARCANE_{event_date_str}_{event_number:02d}"

    for timestamp in merged_df.index:
        for event in catalog_mo:
            if event.begin <= timestamp <= event.end:
                merged_df.loc[timestamp, "event_id"] = event.event_id
                merged_df.loc[timestamp, "proba"] = (
                    "Probability: " + str(event.proba)[:4]
                )

    ######################
    ###### PLOTTING ######
    ######################

    high_res_insitu_df = multi_signal_dataset_high_res.df

    # insert event id into high_res_insitu_df
    high_res_insitu_df["event_id"] = merged_df["event_id"]

    # insert proba into high_res_insitu_df
    high_res_insitu_df["proba"] = merged_df["proba"]

    # ffill the event_id column
    high_res_insitu_df["event_id"] = high_res_insitu_df["event_id"].ffill()

    # ffill the proba column
    high_res_insitu_df["proba"] = high_res_insitu_df["proba"].ffill()

    line_colors = [
        "#000000",
        "#aa00ff",
        "#f97306",
        "#069af3",
    ]

    high_res_insitu_df = high_res_insitu_df.iloc[512:]

    fsize = 14

    variables = ["B", "vt", "np", "tp", "beta"]
    names = {
        "B": "B [nT]",
        "vt": "V [km/s]",
        "np": "N<sub>P</sub> [cm<sup>-3</sup>]",
        "tp": "T<sub>P</sub> [K]",
        "beta": "β",
    }

    fig = make_subplots(rows=len(variables) + 3, cols=1, shared_xaxes=True)

    for i, var in enumerate(variables):
        if var == "B":
            # Plot magnetic field components
            mag_field = {
                "B<sub>TOT</sub>": [
                    col for col in high_res_insitu_df.columns if "bt" in col
                ],
                "B<sub>X</sub>": [
                    col for col in high_res_insitu_df.columns if "bx" in col
                ],
                "B<sub>Y</sub>": [
                    col for col in high_res_insitu_df.columns if "by" in col
                ],
                "B<sub>Z</sub>": [
                    col for col in high_res_insitu_df.columns if "bz" in col
                ],
            }

            n = 0
            for magvar, settings in mag_field.items():
                if settings:
                    fig.add_trace(
                        go.Scatter(
                            x=high_res_insitu_df.index,
                            y=high_res_insitu_df[settings[0]],
                            mode="lines",
                            name=magvar,
                            line=dict(color=line_colors[n]),
                            showlegend=True if i == 0 else False,
                            customdata=high_res_insitu_df[["event_id", "proba"]],
                            hovertemplate="%{y}<br>"
                            + "%{x}<br><br>"
                            + "%{customdata[0]}<br><br>"
                            + "%{customdata[1]}"
                            + "<extra></extra>",
                        ),
                        row=i + 1,
                        col=1,
                    )
                n = n + 1

        else:
            y = [col for col in high_res_insitu_df.columns if var in col]

            fig.add_trace(
                go.Scatter(
                    x=high_res_insitu_df.index,
                    y=high_res_insitu_df[y[0]],
                    mode="lines",
                    name=names[var],
                    line=dict(color="black"),
                    showlegend=False,
                    customdata=high_res_insitu_df["event_id"],
                    hovertemplate="%{y}<br>"
                    + "%{x}<br><br>"
                    + "%{customdata}<extra></extra>",
                ),
                row=i + 1,
                col=1,
            )

        for event in catalog_sheath:
            fig.add_vrect(
                x0=event.begin,
                x1=event.end,
                fillcolor="LightGreen",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=i + 1,
                col=1,
                showlegend=False,
            )

        for event in catalog_mo:
            fig.add_vrect(
                x0=event.begin,
                x1=event.end,
                fillcolor="LightSalmon",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=i + 1,
                col=1,
                showlegend=False,
            )

        fig.update_yaxes(title_text=names.get(var, var), row=i + 1, col=1)
        # Set y-axis limits for beta
        if var == "beta" or var == "np":
            fig.update_yaxes(
                type="log",
                tickvals=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
                row=i + 1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="LightGreen", symbol="square"),
            name="ARCANE Sheath",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="LightSalmon", symbol="square"),
            name="ARCANE MO",
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=merged_df["predicted_value_sheath"].values.reshape(1, -1),
            x=merged_df.index,
            y=["Predicted Value Sheath"],
            zmin=0,
            zmax=1,
            colorscale="Greys",
            showscale=False,
            name="Predicted Value Sheath",
            hovertemplate="Probability: %{z}<br>" + "%{x}<br><br>",
        ),
        row=i + 2,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=merged_df["predicted_value_mo"].values.reshape(1, -1),
            x=merged_df.index,
            y=["Predicted Value MO"],
            zmin=0,
            zmax=1,
            colorscale="Greys",
            showscale=False,
            name="Predicted Value MO",
            hovertemplate="Probability: %{z}<br>" + "%{x}<br><br>",
        ),
        row=i + 3,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=merged_df["predicted_value_threshold"].values.reshape(1, -1),
            x=merged_df.index,
            y=["Predicted Value Threshold"],
            zmin=0,
            zmax=1,
            colorscale="Greys",
            showscale=False,
            name="Predicted Value Threshold",
            hovertemplate="Probability: %{z}<br>" + "%{x}<br><br>",
        ),
        row=i + 4,
        col=1,
    )

    fig.update_yaxes(
        showticklabels=False,
        title_text="ARCANE<br>Probability<br>Sheath",
        row=i + 2,
        col=1,
    )

    fig.update_yaxes(
        showticklabels=False, title_text="ARCANE<br>Probability<br>MO", row=i + 3, col=1
    )

    fig.update_yaxes(
        showticklabels=False, title_text="Threshold<br>Probability", row=i + 4, col=1
    )

    # Hide x-axis tick labels for all but the last 3 subplots
    for i in range(1, len(variables) + 1):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    # Customize x-axis ticks and labels for the last 3 subplots
    for i in range(len(variables), len(variables) + 1):
        fig.update_xaxes(
            tickformat="%b %d",  # Major ticks as dates
            dtick="D1",  # Major ticks every 1 day
            minor=dict(
                dtick=7200000,  # Minor ticks every 2 hours
                ticks="outside",  # Minor tick marks outside
            ),
            ticklabelmode="period",  # Align tick labels with time periods
            ticks="outside",  # Draw major tick marks outside
            showticklabels=True,  # Enable tick labels
            row=i,
            col=1,
            range=[
                datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7),
                datetime.datetime.now(datetime.UTC),
            ],
        )
    # Customize x-axis ticks and labels for the last 3 subplots
    for i in range(len(variables) + 1, len(variables) + 3):
        fig.update_xaxes(
            tickformat="%b %d",  # Major ticks as dates
            dtick="D1",  # Major ticks every 1 day
            minor=dict(
                dtick=7200000,  # Minor ticks every 2 hours
                ticks="outside",  # Minor tick marks outside
            ),
            ticklabelmode="period",  # Align tick labels with time periods
            ticks="outside",  # Draw major tick marks outside
            showticklabels=True,  # Enable tick labels
            row=i,
            col=1,
            range=[
                datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7),
                datetime.datetime.now(datetime.UTC),
            ],
        )

        # Customize y-axis gridlines distance to 0.2
        fig.update_yaxes(
            # gridwidth=0.1,  # Set gridline width
            showgrid=True,  # Enable gridlines
            gridcolor="White",
            row=i,
            col=1,
            range=[0, 0.5],  # Set y-axis range to 0-1
            tickvals=[0.1, 0.2, 0.3, 0.4],
        )

    last_updated = datetime.datetime.now(datetime.UTC).strftime(
        "last update: %Y-%m-%d %H:%M UTC"
    )

    fig.update_layout(
        title="ARCANE - Automatic ICME Detection",
        title_x=0.5,  # Centers the title
        title_font=dict(size=fsize + 6, family="Arial", color="black"),
        annotations=[
            dict(
                x=0,
                y=1.1,
                xref="paper",
                yref="paper",
                text=last_updated,
                showarrow=False,
                font=dict(size=fsize, color="grey"),
                align="left",
            ),
            dict(
                x=0,
                y=-0.1,
                xref="paper",
                yref="paper",
                text="H.T. Rüdisser, G. Nguyen, J. Le Louëdec, C. Möstl<br>Austrian Space Weather Office   helioforecast.space ",
                showarrow=False,
                font=dict(size=fsize, color="grey"),
                align="left",
            ),
            dict(
                x=1,
                y=-0.1,
                xref="paper",
                yref="paper",
                text="(c) GeoSphere Austria",
                showarrow=False,
                font=dict(size=fsize, color="grey"),
                align="right",
            ),
        ],
    )

    fig.update_layout(
        legend=dict(font=dict(size=fsize)),  # Legend font size
        autosize=False,
        width=1400,
        height=1000,
        #        margin=dict(l=50, r=50, b=50, t=50, pad=4),
    )
    # Set font size for all x and y axes
    for i in range(1, len(variables) + 3):  # Adjusted range to include all subplots
        fig.update_xaxes(
            tickfont=dict(size=fsize),  # Tick font size
            title_font=dict(size=fsize),  # Axis label font size
            row=i,
            col=1,
            range=[
                datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7),
                datetime.datetime.now(datetime.UTC),
            ],
        )
        fig.update_yaxes(
            tickfont=dict(size=fsize),  # Tick font size
            title_font=dict(size=fsize),  # Axis label font size
            row=i,
            col=1,
        )

    fig.write_html(cache_path / Path("arcane_plot_multiclass_now.html"))
    fig.write_image(cache_path / Path("arcane_plot_multiclass_now.png"))

    update_event_catalog(
        cache_path / Path("arcane_catalog_now_sheath.csv"), catalog_sheath
    )
    update_event_catalog(cache_path / Path("arcane_catalog_now_mo.csv"), catalog_mo)

    print(f"Event catalog and plots saved in {cache_path.resolve()}")

    os._exit(0)  # Added this line to prevent the script from getting stuck


if __name__ == "__main__":
    main()
