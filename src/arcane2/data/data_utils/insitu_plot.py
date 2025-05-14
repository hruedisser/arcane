import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from .event import EventCatalog


def plotly_plot_insitu(
    data: pd.DataFrame,
    begin: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    variables: list[str] = ["B"],
    source_mag: bool = False,
    source_plasma: bool = False,
    catalogs: list[EventCatalog] = [],
    reference_frame: str = "GSM",
    line_colors: list[str] = ["#000000", "#c20078", "#f97306", "#069af3"],
    line_width: int = 1,
    subdf: pd.DataFrame = None,
    title_text: str | None = None,
    thresh: float | None = None,
):
    """
    Plot insitu data using plotly.

    Args:
        data (pd.DataFrame): Dataframe containing the insitu data.
        begin (datetime.datetime, optional): Start time of the plot. Defaults to None.
        end (datetime.datetime, optional): End time of the plot. Defaults to None.
        variables (list[str], optional): List of variables to plot. Defaults to ["B"].
    """
    filtered_catalogs = []

    if end and begin:
        filtered_data = data[(data.index >= begin) & (data.index <= end)]

        if subdf is not None:
            filtered_subdf = subdf[(subdf.index >= begin) & (subdf.index <= end)]
        else:
            filtered_subdf = None

        for catalog in catalogs:
            filtered_catalog = catalog.filter_catalog(begin, end)
            filtered_catalogs.append(filtered_catalog)
    else:
        filtered_data = data
        if catalog is not None:
            filtered_catalogs = [catalog.event_cat for catalog in catalogs]

    columns = filtered_data.columns

    names = {
        "vt": "V<sub>t</sub> [km/s]",
        "vx": "V<sub>x</sub> [km/s]",
        "vy": "V<sub>y</sub> [km/s]",
        "vz": "V<sub>z</sub> [km/s]",
        "np": "N<sub>p</sub> [cm<sup>-3</sup>]",
        "tp": "T<sub>p</sub> [K]",
        "beta": "\u03B2",
    }

    if thresh is not None and filtered_subdf is not None:
        for col in filtered_subdf.columns:
            if not all(filtered_subdf[col].isin([0, 1])):
                binary_col_name = f"{col}_binary"
                filtered_subdf[binary_col_name] = (
                    (filtered_subdf[col] >= thresh)
                    .astype(int)
                    .resample("10min")
                    .ffill()
                )

    # if "spodify" in self.catalog_name:
    #         positive_labels = labels[labels > self.thresh / 10]
    #         logger.info(
    #             f"Reducing threshold for {self.catalog_name} to {self.thresh/10}"
    #         )
    #     else:

    if filtered_subdf is not None:
        filtered_subdf = filtered_subdf.reindex(sorted(filtered_subdf.columns), axis=1)

    if title_text is None:
        title_text = (
            "Insitu Data - "
            + str(filtered_data.index[0])
            + " to "
            + str(filtered_data.index[-1])
        )

    n_rows = len(variables) + 1 if "Targets" in variables else len(variables)

    specs = []

    # Iterate through variables and construct the specs for each row
    for var in variables:
        if var == "Targets":
            specs.append([{"rowspan": 2}])
            specs.append([None])
        else:
            specs.append([{}])

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=specs,
    )

    for i, var in enumerate(variables):

        if (var == "B") & (reference_frame == "GSM"):

            mag_field = {
                "B<sub>t</sub>": [col for col in columns if "bt" in col]
                + [line_colors[0]],
                "B<sub>x</sub>": [col for col in columns if "bx" in col]
                + [line_colors[1]],
                "B<sub>y</sub>": [col for col in columns if "by" in col]
                + [line_colors[2]],
                "B<sub>z</sub>": [col for col in columns if "bz" in col]
                + [line_colors[3]],
            }

            for magvar, settings in mag_field.items():
                if len(settings) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data.index,
                            y=filtered_data[settings[0]],
                            mode="lines",
                            name=magvar,
                            line=dict(color=settings[1], width=line_width),
                            showlegend=True,
                        ),
                        row=i + 1,
                        col=1,
                    )
            fig.update_yaxes(title_text="B [nT]", row=i + 1, col=1)

        else:
            varname = [col for col in columns if var.lower() in col.lower()]
            if len(varname) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[varname[0]],
                        mode="lines",
                        line=dict(color=line_colors[0], width=line_width),
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=1,
                )
                fig.update_yaxes(title_text=names[var], row=i + 1, col=1)

            else:
                fig.add_trace(
                    go.Heatmap(
                        z=filtered_subdf.values.T,
                        x=filtered_subdf.index,
                        y=filtered_subdf.columns,
                        colorscale="Purples",
                        showscale=False,
                    ),
                    row=i + 1,
                    col=1,
                )

    # Add CME shading if 'CME' is in the flags
    fillcolors = {
        "ICME": "LightSalmon",
        "CME": "LightSalmon",
        "Sheath": "LightGreen",
        "SIR": "LightBlue",
        "Event": "LightBlue",
    }
    for cid, filcat in enumerate(filtered_catalogs):
        for event in tqdm(
            filcat, desc=f"Adding catalog {cid+1}/{len(filtered_catalogs)}"
        ):
            for i in range(len(variables)):
                if i == 0:
                    fig.add_vrect(
                        x0=event.begin,
                        x1=event.end,
                        # annotation_text=event.event_id if i == 0 else None,
                        fillcolor=fillcolors[event.event_type],
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=i + 1,
                        col=1,
                    )
                else:
                    fig.add_vrect(
                        x0=event.begin,
                        x1=event.end,
                        fillcolor=fillcolors[event.event_type],
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=i + 1,
                        col=1,
                    )

    fig.update_layout(
        height=200 * len(variables),
        title_text=title_text,
        xaxis_title=None,  # Do not set global x-axis title
    )

    # Set x-axis title and tick labels only for the bottom subplot
    fig.update_xaxes(title_text="Time", row=len(variables), col=1, showticklabels=True)

    return fig


class InsituPlot:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot(
        self,
        begin: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        variables: list[str] = ["B"],
        source_mag: bool = False,
        source_plasma: bool = False,
        reference_frame: str = "GSM",
        line_colors: list[str] = ["#000000", "#c20078", "#f97306", "#069af3"],
        line_width: int = 1,
    ):
        return plotly_plot_insitu(
            data=self.data,
            begin=begin,
            end=end,
            variables=variables,
            source_mag=source_mag,
            source_plasma=source_plasma,
            catalogs=self.catalogs,
            reference_frame=reference_frame,
            line_colors=line_colors,
            line_width=line_width,
        )
