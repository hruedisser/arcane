import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from py3dcore.methods.heliosat_utils import sanitize_dt

template = "none"
bg_color = "rgba(0, 0,0, 0)"
line_color = "black"
line_colors = ["#c20078", "#f97306", "#069af3", "#000000"]
eventshade = "LightSalmon"

lw_insitu = 2  # linewidth for plotting the in situ data
lw_best = 3  # linewidth for plotting the min(eps) run
lw_mean = 3  # linewidth for plotting the mean run
lw_fitp = 2  # linewidth for plotting the lines where fitting points


def plot_insitu(t_data, b_data, reference_frame="HEEQ"):

    if reference_frame == "HEEQ" or reference_frame == "GSM":
        names = ["B$_X$", "B$_Y$", "B$_Z$"]
    elif reference_frame == "RTN":
        names = ["B$_R$", "B$_T$", "B$_N$"]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True)

    ax.plot(
        t_data, b_data[:, 0], label=names[0], color=line_colors[0], linewidth=lw_insitu
    )
    ax.plot(
        t_data, b_data[:, 1], label=names[1], color=line_colors[1], linewidth=lw_insitu
    )
    ax.plot(
        t_data, b_data[:, 2], label=names[2], color=line_colors[2], linewidth=lw_insitu
    )
    ax.plot(
        t_data,
        np.sqrt(np.sum(b_data**2, axis=1)),
        label="B$_{TOT}$",
        color=line_colors[3],
        linewidth=lw_insitu,
    )
    ax.legend()
    ax.set_ylabel("Magnetic Field [nT]")

    date_form = mdates.DateFormatter("%h %d %H")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=25, ha="right")
    plt.legend(loc="lower right", ncol=2)

    return fig, ax


def plot_results(t_data, b_data, ensemble, t, observer, reference_frame="HEEQ"):

    if reference_frame == "HEEQ" or reference_frame == "GSM":
        names = ["B$_X$", "B$_Y$", "B$_Z$"]
    elif reference_frame == "RTN":
        names = ["B$_R$", "B$_T$", "B$_N$"]

    t_fits = observer[1]
    t_s = observer[2]
    t_e = observer[3]

    plot_start = t_s - datetime.timedelta(hours=6)

    if t == "100":
        plot_end = t_e + datetime.timedelta(hours=6)
    else:
        plot_end = t_fits[-1]

    fig, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

    t_data = [sanitize_dt(_) for _ in t_data]

    t_data = np.array(t_data)
    b_data = np.array(b_data)
    plot_t = t_data[(t_data >= plot_start) & (t_data <= plot_end)]
    plot_b = b_data[(t_data >= plot_start) & (t_data <= plot_end)]

    ax.plot(
        plot_t, plot_b[:, 0], label=names[0], color=line_colors[0], linewidth=lw_insitu
    )
    ax.plot(
        plot_t, plot_b[:, 1], label=names[1], color=line_colors[1], linewidth=lw_insitu
    )
    ax.plot(
        plot_t, plot_b[:, 2], label=names[2], color=line_colors[2], linewidth=lw_insitu
    )
    ax.plot(
        plot_t,
        np.sqrt(np.sum(plot_b**2, axis=1)),
        label="B$_{TOT}$",
        color=line_colors[3],
        linewidth=lw_insitu,
    )

    if t == "100":
        pass
    else:
        plot_start_2 = plot_end
        plot_end_2 = t_e + datetime.timedelta(hours=6)

        plot_t = t_data[(t_data >= plot_start_2) & (t_data <= plot_end_2)]
        plot_b = b_data[(t_data >= plot_start_2) & (t_data <= plot_end_2)]

        # plot the remaining time as dashed lines
        ax.plot(
            plot_t, plot_b[:, 0], color=line_colors[0], linewidth=lw_insitu, ls="dotted"
        )

        ax.plot(
            plot_t, plot_b[:, 1], color=line_colors[1], linewidth=lw_insitu, ls="dotted"
        )

        ax.plot(
            plot_t, plot_b[:, 2], color=line_colors[2], linewidth=lw_insitu, ls="dotted"
        )

        ax.plot(
            plot_t,
            np.sqrt(np.sum(plot_b**2, axis=1)),
            color=line_colors[3],
            linewidth=lw_insitu,
            ls="dotted",
        )

    ax.legend()
    ax.set_ylabel("Magnetic Field [nT]")

    date_form = mdates.DateFormatter("%h %d %H")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=25, ha="right")
    plt.legend(loc="lower right", ncol=2)

    ax.axvline(x=t_s, lw=lw_fitp, alpha=0.75, color="k", ls="-.")
    ax.axvline(x=t_e, lw=lw_fitp, alpha=0.75, color="k", ls="-.")

    for _ in t_fits:
        ax.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")

    ensemble[np.where(ensemble == 0)] = np.nan

    perc = 0.95

    b_s2p = np.nanquantile(ensemble, 0.5 + perc / 2, axis=1)
    b_s2n = np.nanquantile(ensemble, 0.5 - perc / 2, axis=1)

    b_t = np.sqrt(np.sum(ensemble**2, axis=2))
    b_tm = np.nanmean(b_t, axis=1)

    b_ts2p = np.nanquantile(b_t, 0.5 + perc / 2, axis=1)
    b_ts2n = np.nanquantile(b_t, 0.5 - perc / 2, axis=1)

    b_m = np.nanmean(ensemble, axis=1)

    plot_t = t_data[(t_data >= plot_start) & (t_data <= plot_end_2)]

    ax.fill_between(
        plot_t,
        b_s2p[:, 0][(t_data >= plot_start) & (t_data <= plot_end_2)],
        b_s2n[:, 0][(t_data >= plot_start) & (t_data <= plot_end_2)],
        color=line_colors[0],
        alpha=0.25,
    )
    ax.fill_between(
        plot_t,
        b_s2p[:, 1][(t_data >= plot_start) & (t_data <= plot_end_2)],
        b_s2n[:, 1][(t_data >= plot_start) & (t_data <= plot_end_2)],
        color=line_colors[1],
        alpha=0.25,
    )
    ax.fill_between(
        plot_t,
        b_s2p[:, 2][(t_data >= plot_start) & (t_data <= plot_end_2)],
        b_s2n[:, 2][(t_data >= plot_start) & (t_data <= plot_end_2)],
        color=line_colors[2],
        alpha=0.25,
    )
    ax.fill_between(
        plot_t,
        b_ts2p[(t_data >= plot_start) & (t_data <= plot_end_2)],
        b_ts2n[(t_data >= plot_start) & (t_data <= plot_end_2)],
        color=line_colors[3],
        alpha=0.25,
    )

    # set y-axis limits to -+ 300 nT
    ax.set_ylim(-40, 40)

    ax.set_title(f"Prediction after {t} hours")

    fig.tight_layout()

    return fig, ax
