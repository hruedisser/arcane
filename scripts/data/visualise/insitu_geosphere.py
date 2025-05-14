import datetime
import pickle
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Define colors
geo_green = (5 / 255, 46 / 255, 55 / 255)
geo_green2 = (84 / 255, 109 / 255, 115 / 255)
geo_green3 = (157 / 255, 171 / 255, 174 / 255)
geo_green4 = (218 / 255, 224 / 255, 225 / 255)

geo_lime = (191 / 255, 206 / 255, 64 / 255)
geo_lime2 = (210 / 255, 221 / 255, 101 / 255)
geo_lime3 = (229 / 255, 236 / 255, 163 / 255)
geo_lime4 = (246 / 255, 248 / 255, 220 / 255)

# Extended colors
geo_magenta = (140 / 255, 17 / 255, 170 / 255)
geo_purple = (88 / 255, 31 / 255, 128 / 255)
geo_royalpurple = (46 / 255, 0, 159 / 255)
geo_lilac = (88 / 255, 51 / 255, 254 / 255)
geo_lavender = (115 / 255, 102 / 255, 254 / 255)
geo_cornflowerblue = (140 / 255, 153 / 255, 253 / 255)
geo_lightblue = (166 / 255, 204 / 255, 253 / 255)
geo_paleturquoise = (192 / 255, 255 / 255, 252 / 255)
geo_aquamarine = (141 / 255, 243 / 255, 216 / 255)
geo_mintgreen = (90 / 255, 226 / 255, 145 / 255)
geo_grassgreen = (117 / 255, 204 / 255, 65 / 255)
geo_ocher = (191 / 255, 206 / 255, 64 / 255)
geo_yellow = (249 / 255, 242 / 255, 0)
geo_orange = (242 / 255, 151 / 255, 7 / 255)
geo_orangered = (231 / 255, 92 / 255, 19 / 255)
geo_red = (204 / 255, 44 / 255, 1 / 255)

geo_copyright = (154 / 255, 172 / 255, 175 / 255)
geo_axes_title = (5 / 255, 46 / 255, 55 / 255)
geo_raster = (244 / 255, 244 / 255, 244 / 255)

# Define cache path
CACHE_PATH = Path("cache/arcanecore/dataset.pkl")

# Set up matplotlib style
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set_style("ticks")

# Load custom font
font_path = "./source-sans-pro/SourceSansPro-Bold.otf"
prop = fm.FontProperties(fname=font_path)

if __name__ == "__main__":
    # Load dataset from pickle
    with CACHE_PATH.open("rb") as cache_file:
        dataset = pickle.load(cache_file)

    multi_signal_dataset = dataset.dataset

    sample_event = multi_signal_dataset.single_signal_datasets[0].catalog.event_cat[-20]

    catalogs = [
        multi_signal_dataset.single_signal_datasets[0].catalog,
        multi_signal_dataset.single_signal_datasets[1].catalog,
    ]

    begin = sample_event.begin - datetime.timedelta(hours=400)
    end = sample_event.end + datetime.timedelta(hours=100)

    filtered_data = multi_signal_dataset.df[
        (multi_signal_dataset.df.index >= begin)
        & (multi_signal_dataset.df.index <= end)
    ]
    filtered_catalogs = []

    for catalog in catalogs:
        filtered_catalog = catalog.filter_catalog(begin, end)
        filtered_catalogs.append(filtered_catalog)

    title_text = "Automatic Detection of Coronal Mass Ejections using Machine Learning"

    variables = ["B", "vt", "np", "tp", "beta"]
    names = {
        "vt": "V$_{T}$ [km/s]",
        "np": "N$_{P}$ [cm$^{-3}$]",
        "tp": "T$_{P}$ [K]",
        "beta": "\u03B2",
    }

    size = 1920
    my_dpi = 100

    n_rows = len(variables)

    fig, axs = plt.subplots(
        n_rows,
        1,
        figsize=(size / my_dpi, size / my_dpi),
        dpi=my_dpi,
        sharex=True,
        edgecolor=geo_green,
    )
    plt.subplots_adjust(hspace=0.5)

    line_colors = [geo_green, geo_magenta, geo_lime, geo_green2]

    for i, var in enumerate(variables):
        if var == "B":
            # Plotting magnetic field components if in GSM reference frame
            mag_field = {
                "B$_{TOT}$": [col for col in filtered_data.columns if "bt" in col]
                + [line_colors[0]],
                "B$_{X}$": [col for col in filtered_data.columns if "bx" in col]
                + [line_colors[1]],
                "B$_{Y}$": [col for col in filtered_data.columns if "by" in col]
                + [line_colors[2]],
                "B$_{Z}$": [col for col in filtered_data.columns if "bz" in col]
                + [line_colors[3]],
            }

            for magvar, settings in mag_field.items():
                if settings:
                    axs[i].plot(
                        filtered_data.index,
                        filtered_data[settings[0]],
                        color=settings[1],
                        label=magvar,
                    )
            axs[i].set_ylabel(
                "Magnetic Field [nT]", fontproperties=prop, fontsize=16, color="#052E37"
            )
            axs[i].set_ylim(-20, 30)  # Set y-axis range from -20 to 20

        else:
            varname = [
                col for col in filtered_data.columns if var.lower() in col.lower()
            ]
            if varname:
                axs[i].plot(
                    filtered_data.index,
                    filtered_data[varname[0]],
                    color=line_colors[0],
                    label=names.get(var, var),
                )

            # Set y-axis labels
            axs[i].set_ylabel(
                names.get(var, var), fontproperties=prop, fontsize=16, color="#052E37"
            )

        # Remove legends from all but the first subplot
        if i == 0:
            axs[i].legend(
                loc="lower left",
                frameon=False,
                ncol=len(mag_field),
            )

    # Adding vertical rectangles for events
    for cid, filcat in enumerate(filtered_catalogs):
        for event in tqdm(
            filcat, desc=f"Adding catalog {cid + 1}/{len(filtered_catalogs)}"
        ):
            for i in range(len(variables)):
                axs[i].axvspan(event.begin, event.end, color=geo_lime4)  # , alpha=0.2)

    # Add title as an annotation
    axs[0].annotate(
        title_text,
        xy=(0.5, 1.15),  # position the text at the top center
        xycoords="axes fraction",
        fontsize=30,
        fontweight="bold",
        ha="center",
        color="#052E37",
        fontproperties=prop,
    )

    axs[-1].set_xlabel("", fontproperties=prop, fontsize=16, color="#052E37")

    # Set ticks font properties
    for ax in axs:
        ax.tick_params(axis="both", labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)
            label.set_color("#052E37")

    plt.tight_layout()

    # Add copyright notice
    plt.figtext(
        0.99,
        0.01,
        "© GeoSphere Austria",
        horizontalalignment="right",
        fontsize=16,
        color="#9AACAF",
        fontproperties=prop,
    )

    plt.savefig("automatic_insitu_cme_detection_en.png", dpi=my_dpi)

    fig, axs = plt.subplots(
        n_rows,
        1,
        figsize=(size / my_dpi, size / my_dpi),
        dpi=my_dpi,
        sharex=True,
        edgecolor=geo_green,
    )
    plt.subplots_adjust(hspace=0.5)
    title_text = "Automatische Detektion von Sonnenstürmen mit Machine Learning"

    line_colors = [geo_green, geo_magenta, geo_lime, geo_green2]

    for i, var in enumerate(variables):
        if var == "B":
            # Plotting magnetic field components if in GSM reference frame
            mag_field = {
                "B$_{TOT}$": [col for col in filtered_data.columns if "bt" in col]
                + [line_colors[0]],
                "B$_{X}$": [col for col in filtered_data.columns if "bx" in col]
                + [line_colors[1]],
                "B$_{Y}$": [col for col in filtered_data.columns if "by" in col]
                + [line_colors[2]],
                "B$_{Z}$": [col for col in filtered_data.columns if "bz" in col]
                + [line_colors[3]],
            }

            for magvar, settings in mag_field.items():
                if settings:
                    axs[i].plot(
                        filtered_data.index,
                        filtered_data[settings[0]],
                        color=settings[1],
                        label=magvar,
                    )
            axs[i].set_ylabel(
                "Magnetfeld [nT]", fontproperties=prop, fontsize=16, color="#052E37"
            )
            axs[i].set_ylim(-20, 30)  # Set y-axis range from -20 to 20

        else:
            varname = [
                col for col in filtered_data.columns if var.lower() in col.lower()
            ]
            if varname:
                axs[i].plot(
                    filtered_data.index,
                    filtered_data[varname[0]],
                    color=line_colors[0],
                    label=names.get(var, var),
                )

            # Set y-axis labels
            axs[i].set_ylabel(
                names.get(var, var), fontproperties=prop, fontsize=16, color="#052E37"
            )

        # Remove legends from all but the first subplot
        if i == 0:
            axs[i].legend(
                loc="lower left",
                frameon=False,
                ncol=len(mag_field),
            )

    # Adding vertical rectangles for events
    for cid, filcat in enumerate(filtered_catalogs):
        for event in tqdm(
            filcat, desc=f"Adding catalog {cid + 1}/{len(filtered_catalogs)}"
        ):
            for i in range(len(variables)):
                axs[i].axvspan(event.begin, event.end, color=geo_lime4)  # , alpha=0.2)

    # Add title as an annotation
    axs[0].annotate(
        title_text,
        xy=(0.5, 1.15),  # position the text at the top center
        xycoords="axes fraction",
        fontsize=30,
        fontweight="bold",
        ha="center",
        color="#052E37",
        fontproperties=prop,
    )

    axs[-1].set_xlabel("", fontproperties=prop, fontsize=16, color="#052E37")

    # Set ticks font properties
    for ax in axs:
        ax.tick_params(axis="both", labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)
            label.set_color("#052E37")

    plt.tight_layout()

    # Add copyright notice
    plt.figtext(
        0.99,
        0.01,
        "© GeoSphere Austria",
        horizontalalignment="right",
        fontsize=16,
        color="#9AACAF",
        fontproperties=prop,
    )

    plt.savefig("automatic_insitu_cme_detection_de.png", dpi=my_dpi)
