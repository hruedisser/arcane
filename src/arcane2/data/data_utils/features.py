import numpy as np
import pandas as pd
import scipy.constants as constants
from tqdm import tqdm


def computeBetawiki(data):
    """
    compute Beta according to wiki
    """
    try:
        data["beta"] = (
            1e6
            * data["np"]
            * constants.Boltzmann
            * data["tp"]
            / (np.square(1e-9 * data["bt"]) / (2 * constants.mu_0))
        )
    except KeyError:
        print("KeyError")

    return data


def computePdyn(data):
    """
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    the function assume data already has ['Np','V'] features
    """
    try:
        data["pdyn"] = 1e12 * constants.m_p * data["np"] * data["vt"] ** 2
    except KeyError:
        print("Error computing Pdyn, V or Np might not be loaded " "in dataframe")
    return data


def computeTexrat(data):
    """
    compute the ratio of Tp/Tex
    """
    try:
        data["texrat"] = data["tp"] * 1e-3 / (np.square(0.031 * data["vt"] - 5.1))
    except KeyError:
        print("Error computing Texrat")

    return data


def computeQuartiles(data, shifthours=15):
    shifted_bz = data["bz"].shift(-shifthours, freq="H")
    shifted_bt = data["bt"].shift(-shifthours, freq="H")

    data["bz_q25"] = shifted_bz.rolling(window=shifthours).quantile(0.25)
    data["bz_q50"] = shifted_bz.rolling(window=shifthours).quantile(0.50)
    data["bz_q75"] = shifted_bz.rolling(window=shifthours).quantile(0.75)

    data["bt_q25"] = shifted_bt.rolling(window=shifthours).quantile(0.25)
    data["bt_q50"] = shifted_bt.rolling(window=shifthours).quantile(0.50)
    data["bt_q75"] = shifted_bt.rolling(window=shifthours).quantile(0.75)

    return data


def computeCumsumMax(data, shifthours=15):

    data["bz_negcumsum"] = 0
    data["bt_max"] = np.nan

    for current_time in tqdm(data.index):
        time_window_end = current_time + pd.Timedelta(hours=shifthours)

        future_bz = data.loc[current_time:time_window_end, "bz"]
        future_bt = data.loc[current_time:time_window_end, "bt"]

        data.loc[current_time, "bz_negcumsum"] = float(future_bz[future_bz < 0].sum())
        if len(future_bt) > shifthours:
            data.loc[current_time, "bt_max"] = float(future_bt.max())

    return data


def computekeyparams(data, key_components, shifthours=10):
    """
    compute the key parameters for the data
    """

    for key_component in key_components:
        data[key_component] = np.nan

    for current_time in tqdm(data.index):
        time_window_end = current_time + pd.Timedelta(hours=shifthours)

        future_bz = data.loc[current_time:time_window_end, "bz"]
        future_bt = data.loc[current_time:time_window_end, "bt"]

        if len(future_bt) > shifthours:

            if "bz_min" in key_components:
                data.loc[current_time, "bz_min"] = float(future_bz.min())

            if "bz_mean" in key_components:
                data.loc[current_time, "bz_mean"] = float(future_bz.mean())

            if "bt_max" in key_components:
                data.loc[current_time, "bt_max"] = float(future_bt.max())

            if "bt_mean" in key_components:
                data.loc[current_time, "bt_mean"] = float(future_bt.mean())

            if "bz_std" in key_components:
                data.loc[current_time, "bz_std"] = float(future_bz.std())

            if "bt_std" in key_components:
                data.loc[current_time, "bt_std"] = float(future_bt.std())

    return data
