import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

geo_cornflowerblue = "dodgerblue"
geo_lime = "gold"
geo_magenta = "firebrick"

# write all output to a file
from pathlib import Path

from .data.rtsw import functions_noaa as fa
from .data.rtsw import position_frame_transforms as pos_transform

# sys.stdout = open("omni_shift_analysis.txt", "w")

analysis = 2

# 0: Analzye the timeshift given in the OMNI data
# 1: Calculate the differences and analyze precise timeshift for each event
# 2: Calculate the differences and analyze precise timeshift for each event, using EEDavies files

if analysis == 0:

    print("Analyzing the timeshift given in the OMNI data...")

    years = [
        1998,
        1999,
        2000,
        2001,
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
    ]

    # create a datetime index for the entire range of years with a frequency of 1 minute
    datetime_index = pd.date_range(start="1998-01-01", end="2026-12-31", freq="1min")
    dataframe = pd.DataFrame(index=datetime_index)

    dataframe["timeshift"] = None

    for year in years:

        print(f"Processing year: {year}")

        link = (
            f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_min{year}.asc"
        )

        df = pd.read_csv(link, delim_whitespace=True, header=None, skiprows=1)

        year_col = df[0]
        day_col = df[1]
        hour_col = df[2]
        minute_col = df[3]

        # make datetime column from year, day, hour, minute
        datetime_col = pd.to_datetime(
            year_col.astype(str)
            + "-"
            + day_col.astype(str)
            + " "
            + hour_col.astype(str)
            + ":"
            + minute_col.astype(str),
            format="%Y-%j %H:%M",
        )

        # set the datetime column as index
        df.set_index(datetime_col, inplace=True)

        timeshift_col = df[9]

        # set entries in the dataframe to NaN where timeshift is 999999
        timeshift_col[timeshift_col == 999999] = None

        # set the timeshift column as the value for the corresponding datetime index
        dataframe.loc[datetime_col, "timeshift"] = timeshift_col

    # drop rows with NaN values in the timeshift column
    dataframe = dataframe.dropna()

    # check if theres still any values "999999" in the timeshift column
    if (dataframe["timeshift"] == 999999).any():
        print(
            "There are still values '999999' in the timeshift column. Please check the data."
        )
    else:
        print("All values '999999' have been replaced with NaN.")

    print(f"Mean timeshift: {dataframe['timeshift'].mean()/60/60:.2f} hours")
    print(f"Max timeshift: {dataframe['timeshift'].max()/60/60:.2f} hours")
    print(f"Min timeshift: {dataframe['timeshift'].min()/60/60:.2f} hours")
    print(
        f"Mean absolute timeshift: {dataframe['timeshift'].abs().mean()/60/60:.2f} hours"
    )

    spacecraft_specific_years = [2016, 2017, 2018, 2019]
    spacecraft_all = ["ace", "wind", "dscov"]

    dataframe_spacecraft_specific = pd.DataFrame(
        index=datetime_index, columns=spacecraft_all
    )

    for year in spacecraft_specific_years:
        for spacecraft in spacecraft_all:
            print(f"Processing year: {year}, spacecraft: {spacecraft}")

            link = f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/sc_specific/{spacecraft}_min_b{year}.txt"

            df = pd.read_csv(link, delim_whitespace=True, header=None, skiprows=1)

            year_col = df[0]
            day_col = df[1]
            hour_col = df[2]
            minute_col = df[3]

            # make datetime column from year, day, hour, minute
            datetime_col = pd.to_datetime(
                year_col.astype(str)
                + "-"
                + day_col.astype(str)
                + " "
                + hour_col.astype(str)
                + ":"
                + minute_col.astype(str),
                format="%Y-%j %H:%M",
            )

            # set the datetime column as index
            df.set_index(datetime_col, inplace=True)

            timeshift_col = df[7]

            # set entries in the dataframe to NaN where timeshift is 999999
            timeshift_col[timeshift_col == 999999] = None

            # set the timeshift column as the value for the corresponding datetime index
            dataframe_spacecraft_specific.loc[datetime_col, spacecraft] = timeshift_col

    # drop rows with NaN values in the timeshift column
    dataframe_spacecraft_specific = dataframe_spacecraft_specific.dropna()

    dataframe_spacecraft_specific["dscovr-wind"] = (
        dataframe_spacecraft_specific["dscov"] - dataframe_spacecraft_specific["wind"]
    )
    dataframe_spacecraft_specific["dscovr-ace"] = (
        dataframe_spacecraft_specific["dscov"] - dataframe_spacecraft_specific["ace"]
    )
    dataframe_spacecraft_specific["wind-ace"] = (
        dataframe_spacecraft_specific["wind"] - dataframe_spacecraft_specific["ace"]
    )

    print(f"Mean timeshift for spacecraft-specific data:")
    for spacecraft in spacecraft_all:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].mean()/60/60:.2f} hours"
        )

    print(f"Max timeshift for spacecraft-specific data:")
    for spacecraft in spacecraft_all:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].max()/60/60:.2f} hours"
        )

    print(f"Min timeshift for spacecraft-specific data:")
    for spacecraft in spacecraft_all:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].min()/60/60:.2f} hours"
        )

    print(f"Mean absolute timeshift for spacecraft-specific data:")
    for spacecraft in spacecraft_all:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].abs().mean()/60/60:.2f} hours"
        )

    print(f"Mean timeshift for spacecraft-specific differences:")
    for spacecraft in ["dscovr-wind", "dscovr-ace", "wind-ace"]:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].mean()/60/60:.2f} hours"
        )

    print(f"Max timeshift for spacecraft-specific differences:")
    for spacecraft in ["dscovr-wind", "dscovr-ace", "wind-ace"]:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].max()/60/60:.2f} hours"
        )

    print(f"Min timeshift for spacecraft-specific differences:")
    for spacecraft in ["dscovr-wind", "dscovr-ace", "wind-ace"]:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].min()/60/60:.2f} hours"
        )

    print(f"Mean absolute timeshift for spacecraft-specific differences:")
    for spacecraft in ["dscovr-wind", "dscovr-ace", "wind-ace"]:
        print(
            f"{spacecraft}: {dataframe_spacecraft_specific[spacecraft].abs().mean()/60/60:.2f} hours"
        )


elif analysis == 1:

    file_dir = Path(__file__).resolve()
    data_dir = file_dir.parent.parent / "data"
    kernels_path = data_dir / "kernels"

    print(
        "Calculating the differences and analyzing precise timeshift for each event..."
    )

    if Path(data_dir / "omni_shift_analysis_results.csv").exists():
        print("Results already exist. Skipping analysis.")
        results_df = pd.read_csv(data_dir / "omni_shift_analysis_results.csv")

    else:

        print("Loading the ICMECAT catalog...")
        # Load the helioforecast catalog
        url = (
            "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv"
        )
        icmecat = pd.read_csv(url)

        # filter by spacecraft
        isc = icmecat.loc[:, "sc_insitu"]
        iind = np.where(isc == "Wind")[0]

        print("Loading the in situ rtsw data...")
        # Load in situ rtsw data for getting the rtsw positions
        insitu_data_path = Path(data_dir / "noaa_archive_gsm.p")

        [data, _] = pickle.load(open(insitu_data_path, "rb"))

        dataframe = pd.DataFrame(data)
        dataframe.set_index("time", inplace=True)
        dataframe.index.name = None
        dataframe.index = dataframe.index.tz_localize(None)

        print("Loading WIND data...")
        [wind_data, _] = pickle.load(
            open(data_dir / "wind_pos_HEEQ_19941113_20250731.p", "rb")
        )
        df_wind_pos_HEEQ = pd.DataFrame(wind_data)
        df_wind_pos_HEEQ.set_index("time", inplace=True)
        df_wind_pos_HEEQ.index.name = None
        df_wind_pos_HEEQ.index = df_wind_pos_HEEQ.index.tz_localize(None)

        # get ACE positions from url
        print("Loading the ACE positions...")
        file_df = pd.read_csv(
            "https://izw1.caltech.edu/ACE/ASC/DATA/pos_att/ACE_GSE_position.txt",
            delimiter="\t",
        )

        # Create a datetime column using vectorized operations
        file_df["time"] = (
            pd.to_datetime(file_df["Year"], format="%Y")
            + pd.to_timedelta(file_df["DOY"] - 1, unit="D")
            + pd.to_timedelta(file_df["Secofday"], unit="s")
        )

        # Apply position transformations
        df_ace_pos_HEE = pos_transform.GSE_to_HEE(file_df)
        df_ace_pos_HEEQ = pos_transform.HEE_to_HEEQ(df_ace_pos_HEE)

        # Set index and resample
        df_ace_pos = (
            df_ace_pos_HEEQ.set_index("time")
            .resample("1min")
            .interpolate(method="linear")
            .reset_index(drop=False)
        )

        start_timestamp = datetime.datetime(2016, 7, 26)
        # today minus 1 day
        end_timestamp = datetime.datetime.now() - datetime.timedelta(days=1)

        # get DSCOVR positions from kernels
        print("Loading the DSCOVR positions...")

        df_dscovr = fa.get_dscovrpositions(
            start_timestamp, end_timestamp, kernels_path=kernels_path
        )

        # Apply position transformations
        df_dscovr_pos_HEE = pos_transform.GSE_to_HEE(df_dscovr)
        df_dscovr_pos_HEEQ = pos_transform.HEE_to_HEEQ(df_dscovr_pos_HEE)

        # Set index and resample
        df_dscovr_pos = (
            df_dscovr_pos_HEEQ.set_index("time")
            .resample("1min")
            .interpolate(method="linear")
            .reset_index(drop=False)
        )

        df_dscovr_pos = df_dscovr_pos.set_index("time")
        df_ace_pos = df_ace_pos.set_index("time")

        # Create DataFrame for storing the positions for each spacecraft and each event

        nr_wind_events = len(iind)
        print(f"Number of Wind events: {nr_wind_events}")

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(
            np.nan,
            index=range(nr_wind_events),
            columns=[
                "icme_start_time",
                "mo_start_time",
                "mo_end_time",
                "mo_sc_heliodistance",
                "mo_sc_long_heeq",
                "mo_sc_lat_heeq",
                "icme_speed_mean",
                "mo_speed_mean",
                "rtsw_r",
                "rtsw_lon",
                "rtsw_lat",
                "ace_r",
                "ace_lon",
                "ace_lat",
                "dscovr_r",
                "dscovr_lon",
                "dscovr_lat",
                "wind_r",
                "wind_lon",
                "wind_lat",
            ],
        )

        for event_nr in range(nr_wind_events):
            print(f"Processing event {event_nr + 1}/{nr_wind_events}")
            i = iind[event_nr]
            icme_start_time = pd.to_datetime(
                icmecat.loc[i, "icme_start_time"]
            ).tz_localize(None)
            mo_start_time = pd.to_datetime(icmecat.loc[i, "mo_start_time"]).tz_localize(
                None
            )
            mo_end_time = pd.to_datetime(icmecat.loc[i, "mo_end_time"]).tz_localize(
                None
            )

            # Get the spacecraft heliodistance and longitude/latitude in HEEQ
            mo_sc_heliodistance = icmecat.loc[i, "mo_sc_heliodistance"]
            mo_sc_longitude_heeq = icmecat.loc[i, "mo_sc_long_heeq"]
            mo_sc_latitude_heeq = icmecat.loc[i, "mo_sc_lat_heeq"]

            # Calculate the mean speed of the ICME and MO
            icme_speed_mean = icmecat.loc[event_nr, "icme_speed_mean"]
            mo_speed_mean = icmecat.loc[event_nr, "mo_speed_mean"]

            # fill the results DataFrame
            results_df.loc[event_nr, "icme_start_time"] = icme_start_time
            results_df.loc[event_nr, "mo_start_time"] = mo_start_time
            results_df.loc[event_nr, "mo_end_time"] = mo_end_time
            results_df.loc[event_nr, "mo_sc_heliodistance"] = mo_sc_heliodistance
            results_df.loc[event_nr, "mo_sc_long_heeq"] = mo_sc_longitude_heeq
            results_df.loc[event_nr, "mo_sc_lat_heeq"] = mo_sc_latitude_heeq
            results_df.loc[event_nr, "icme_speed_mean"] = icme_speed_mean
            results_df.loc[event_nr, "mo_speed_mean"] = mo_speed_mean

            dscovr_nearest_time = df_dscovr_pos.index.get_indexer(
                [icme_start_time], method="nearest", tolerance=pd.Timedelta("1h")
            )
            if dscovr_nearest_time[0] == -1:
                print(f"No close timestamp found for {icme_start_time} in DSCOVR data")
                results_df.loc[event_nr, "dscovr_r"] = np.nan
                results_df.loc[event_nr, "dscovr_lon"] = np.nan
                results_df.loc[event_nr, "dscovr_lat"] = np.nan
            else:
                dscovr_row = df_dscovr_pos.iloc[dscovr_nearest_time[0]]
                results_df.loc[event_nr, "dscovr_r"] = dscovr_row["r"]
                results_df.loc[event_nr, "dscovr_lon"] = dscovr_row["lon"]
                results_df.loc[event_nr, "dscovr_lat"] = dscovr_row["lat"]

            ace_nearest_time = df_ace_pos.index.get_indexer(
                [icme_start_time], method="nearest", tolerance=pd.Timedelta("1h")
            )
            if ace_nearest_time[0] == -1:
                print(f"No close timestamp found for {icme_start_time} in ACE data")
                results_df.loc[event_nr, "ace_r"] = np.nan
                results_df.loc[event_nr, "ace_lon"] = np.nan
                results_df.loc[event_nr, "ace_lat"] = np.nan
            else:
                ace_row = df_ace_pos.iloc[ace_nearest_time[0]]
                results_df.loc[event_nr, "ace_r"] = ace_row["r"]
                results_df.loc[event_nr, "ace_lon"] = ace_row["lon"]
                results_df.loc[event_nr, "ace_lat"] = ace_row["lat"]

            rtsw_nearest_time = dataframe.index.get_indexer(
                [icme_start_time], method="nearest", tolerance=pd.Timedelta("1h")
            )
            if rtsw_nearest_time[0] == -1:
                print(f"No close timestamp found for {icme_start_time} in RTSW data")
                results_df.loc[event_nr, "rtsw_r"] = np.nan
                results_df.loc[event_nr, "rtsw_lon"] = np.nan
                results_df.loc[event_nr, "rtsw_lat"] = np.nan
            else:
                rtsw_row = dataframe.iloc[rtsw_nearest_time[0]]
                results_df.loc[event_nr, "rtsw_r"] = rtsw_row["r"]
                results_df.loc[event_nr, "rtsw_lon"] = rtsw_row["lon"]
                results_df.loc[event_nr, "rtsw_lat"] = rtsw_row["lat"]

            wind_nearest_time = df_wind_pos_HEEQ.index.get_indexer(
                [icme_start_time], method="nearest", tolerance=pd.Timedelta("1h")
            )
            if wind_nearest_time[0] == -1:
                print(f"No close timestamp found for {icme_start_time} in WIND data")
                results_df.loc[event_nr, "wind_r"] = np.nan
                results_df.loc[event_nr, "wind_lon"] = np.nan
                results_df.loc[event_nr, "wind_lat"] = np.nan
            else:
                wind_row = df_wind_pos_HEEQ.iloc[wind_nearest_time[0]]
                results_df.loc[event_nr, "wind_r"] = wind_row["r"]
                results_df.loc[event_nr, "wind_lon"] = wind_row["lon"]
                results_df.loc[event_nr, "wind_lat"] = wind_row["lat"]

        # save the results DataFrame to a CSV file
        results_df.to_csv(data_dir / "omni_shift_analysis_results.csv", index=False)

        print("Results saved to omni_shift_analysis_results.csv")

    # delete all rows where rtsw_r is NaN
    results_df = results_df.dropna(subset=["rtsw_r"])

    # calculate the radial distances
    results_df["radial_distance_rtsw_windcat"] = (
        results_df["rtsw_r"] - results_df["mo_sc_heliodistance"]
    ) * 1.495978707e8  # convert to km
    results_df["radial_distance_ace_windcat"] = (
        results_df["ace_r"] - results_df["mo_sc_heliodistance"]
    ) * 1.495978707e8  # convert to km
    results_df["radial_distance_dscovr_windcat"] = (
        results_df["dscovr_r"] - results_df["mo_sc_heliodistance"]
    ) * 1.495978707e8  # convert to km
    results_df["radial_distance_dscovr_ace"] = (
        results_df["dscovr_r"] - results_df["ace_r"]
    ) * 1.495978707e8  # convert to km
    results_df["radial_distance_rtsw_wind"] = (
        results_df["rtsw_r"] - results_df["wind_r"]
    ) * 1.495978707e8  # convert to km
    results_df["radial_distance_ace_wind"] = (
        results_df["ace_r"] - results_df["wind_r"]
    ) * 1.495978707e8  # convert to km
    results_df["radial_distance_dscovr_wind"] = (
        results_df["dscovr_r"] - results_df["wind_r"]
    ) * 1.495978707e8  # convert to km

    # convert to cartesian coordinates to calculate the absolute distances between the spacecraft
    results_df["rtsw_x"] = (
        results_df["rtsw_r"]
        * np.cos(np.radians(results_df["rtsw_lat"]))
        * np.cos(np.radians(results_df["rtsw_lon"]))
    )
    results_df["rtsw_y"] = (
        results_df["rtsw_r"]
        * np.cos(np.radians(results_df["rtsw_lat"]))
        * np.sin(np.radians(results_df["rtsw_lon"]))
    )
    results_df["rtsw_z"] = results_df["rtsw_r"] * np.sin(
        np.radians(results_df["rtsw_lat"])
    )

    results_df["ace_x"] = (
        results_df["ace_r"]
        * np.cos(np.radians(results_df["ace_lat"]))
        * np.cos(np.radians(results_df["ace_lon"]))
    )
    results_df["ace_y"] = (
        results_df["ace_r"]
        * np.cos(np.radians(results_df["ace_lat"]))
        * np.sin(np.radians(results_df["ace_lon"]))
    )
    results_df["ace_z"] = results_df["ace_r"] * np.sin(
        np.radians(results_df["ace_lat"])
    )

    results_df["dscovr_x"] = (
        results_df["dscovr_r"]
        * np.cos(np.radians(results_df["dscovr_lat"]))
        * np.cos(np.radians(results_df["dscovr_lon"]))
    )
    results_df["dscovr_y"] = (
        results_df["dscovr_r"]
        * np.cos(np.radians(results_df["dscovr_lat"]))
        * np.sin(np.radians(results_df["dscovr_lon"]))
    )
    results_df["dscovr_z"] = results_df["dscovr_r"] * np.sin(
        np.radians(results_df["dscovr_lat"])
    )

    results_df["windcat_x"] = (
        results_df["mo_sc_heliodistance"]
        * np.cos(np.radians(results_df["mo_sc_lat_heeq"]))
        * np.cos(np.radians(results_df["mo_sc_long_heeq"]))
    )
    results_df["windcat_y"] = (
        results_df["mo_sc_heliodistance"]
        * np.cos(np.radians(results_df["mo_sc_lat_heeq"]))
        * np.sin(np.radians(results_df["mo_sc_long_heeq"]))
    )
    results_df["windcat_z"] = results_df["mo_sc_heliodistance"] * np.sin(
        np.radians(results_df["mo_sc_lat_heeq"])
    )

    results_df["wind_x"] = (
        results_df["wind_r"]
        * np.cos(np.radians(results_df["wind_lat"]))
        * np.cos(np.radians(results_df["wind_lon"]))
    )
    results_df["wind_y"] = (
        results_df["wind_r"]
        * np.cos(np.radians(results_df["wind_lat"]))
        * np.sin(np.radians(results_df["wind_lon"]))
    )
    results_df["wind_z"] = results_df["wind_r"] * np.sin(
        np.radians(results_df["wind_lat"])
    )

    # calculate the absolute distances between the spacecraft
    results_df["distance_rtsw_wind"] = (
        np.sqrt(
            (results_df["rtsw_x"] - results_df["wind_x"]) ** 2
            + (results_df["rtsw_y"] - results_df["wind_y"]) ** 2
            + (results_df["rtsw_z"] - results_df["wind_z"]) ** 2
        )
        * 1.495978707e8
    )  # convert to km

    results_df["distance_ace_wind"] = (
        np.sqrt(
            (results_df["ace_x"] - results_df["wind_x"]) ** 2
            + (results_df["ace_y"] - results_df["wind_y"]) ** 2
            + (results_df["ace_z"] - results_df["wind_z"]) ** 2
        )
        * 1.495978707e8
    )  # convert to km

    results_df["distance_dscovr_wind"] = (
        np.sqrt(
            (results_df["dscovr_x"] - results_df["wind_x"]) ** 2
            + (results_df["dscovr_y"] - results_df["wind_y"]) ** 2
            + (results_df["dscovr_z"] - results_df["wind_z"]) ** 2
        )
        * 1.495978707e8
    )  # convert to km

    results_df["distance_dscovr_ace"] = (
        np.sqrt(
            (results_df["dscovr_x"] - results_df["ace_x"]) ** 2
            + (results_df["dscovr_y"] - results_df["ace_y"]) ** 2
            + (results_df["dscovr_z"] - results_df["ace_z"]) ** 2
        )
        * 1.495978707e8
    )  # convert to km

    # calculate the time in minutes it takes to go from one spacecraft to the other based on icme_speed_mean, which is in km/s
    results_df["time_rtsw_wind"] = results_df["distance_rtsw_wind"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes
    results_df["time_ace_wind"] = results_df["distance_ace_wind"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes
    results_df["time_dscovr_wind"] = results_df["distance_dscovr_wind"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes
    results_df["time_dscovr_ace"] = results_df["distance_dscovr_ace"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes

    # calculate the time in minutes it takes to go from one spacecraft to the other based on mo_speed_mean, which is in km/s
    results_df["time_rtsw_wind_mo"] = results_df["distance_rtsw_wind"] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes
    results_df["time_ace_wind_mo"] = results_df["distance_ace_wind"] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes
    results_df["time_dscovr_wind_mo"] = results_df["distance_dscovr_wind"] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes
    results_df["time_dscovr_ace_mo"] = results_df["distance_dscovr_ace"] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes

    # calculate the same time but for the radial distances
    results_df["time_radial_rtsw_wind"] = results_df["radial_distance_rtsw_wind"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes
    results_df["time_radial_ace_wind"] = results_df["radial_distance_ace_wind"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes
    results_df["time_radial_dscovr_wind"] = results_df[
        "radial_distance_dscovr_wind"
    ] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes
    results_df["time_radial_dscovr_ace"] = results_df["radial_distance_dscovr_ace"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes

    # calculate the same time but for the radial distances based on mo_speed_mean
    results_df["time_radial_rtsw_wind_mo"] = results_df["radial_distance_rtsw_wind"] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes
    results_df["time_radial_ace_wind_mo"] = results_df["radial_distance_ace_wind"] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes
    results_df["time_radial_dscovr_wind_mo"] = results_df[
        "radial_distance_dscovr_wind"
    ] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes
    results_df["time_radial_dscovr_ace_mo"] = results_df[
        "radial_distance_dscovr_ace"
    ] / (
        results_df["mo_speed_mean"] * 60
    )  # in minutes

    # save the results DataFrame to a CSV file
    results_df.to_csv(
        data_dir / "omni_shift_analysis_results_processed.csv", index=False
    )

elif analysis == 2:

    print(
        "Calculating the differences and analyzing precise timeshift for each event, using EEDavies files, so in GSE.."
    )

    file_dir = Path(__file__).resolve()
    data_dir = file_dir.parent.parent / "data"
    kernels_path = data_dir / "kernels"
    plot_dir = file_dir.parent.parent / "plots"

    print(
        "Calculating the differences and analyzing precise timeshift for each event..."
    )

    print("Creating a DataFrame to store the positions of all three spacecraft...")

    print("Loading the ICMECAT catalog...")
    # Load the helioforecast catalog
    url = "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv"
    icmecat = pd.read_csv(url)

    # filter by spacecraft
    isc = icmecat.loc[:, "sc_insitu"]
    iind = np.where(isc == "Wind")[0]

    print("Loading Wind data...")
    [wind_data, _] = pickle.load(
        open(data_dir / "wind_pos_GSE_19941113_20250731.p", "rb")
    )
    df_wind_pos_GSE = pd.DataFrame(wind_data)
    df_wind_pos_GSE.set_index("time", inplace=True)
    df_wind_pos_GSE.index.name = None
    df_wind_pos_GSE.index = df_wind_pos_GSE.index.tz_localize(None)

    # if an z value in 2019 is below -10000.0, then it is a bad value
    mask_2019_bad = (
        (df_wind_pos_GSE.index.year == 2019)
        & (df_wind_pos_GSE["z"] < -10000.0)
        & (df_wind_pos_GSE.index.month == 3)
    )

    df_wind_pos_GSE = df_wind_pos_GSE[~mask_2019_bad]

    if Path(data_dir / "ace_pos_GSE_19970902_20250731_cleaned.p").exists():
        print("ACE positions already cleaned. Loading cleaned data...")
        ace_data = pickle.load(
            open(data_dir / "ace_pos_GSE_19970902_20250731_cleaned.p", "rb")
        )
        df_ace_pos_GSE = pd.DataFrame(ace_data)
    else:

        print("Loading the ACE positions from mag file...")
        [ace_data, _] = pickle.load(
            open(data_dir / "ace_pos_GSE_19970902_20250731.p", "rb")
        )
        df_ace_pos_GSE = pd.DataFrame(ace_data)
        df_ace_pos_GSE.set_index("time", inplace=True)
        df_ace_pos_GSE.index.name = None
        df_ace_pos_GSE.index = df_ace_pos_GSE.index.tz_localize(None)

        # save the cleaned ACE positions to a file
        df_ace_pos_GSE.to_pickle(data_dir / "ace_pos_GSE_19970902_20250731_cleaned.p")

    print("Loading the DSCOVR positions...")
    [dscovr_data, _] = pickle.load(
        open(data_dir / "dscovr_pos_GSE_20160726_20250731.p", "rb")
    )
    df_dscovr_pos_GSE = pd.DataFrame(dscovr_data)
    df_dscovr_pos_GSE.set_index("time", inplace=True)
    df_dscovr_pos_GSE.index.name = None
    df_dscovr_pos_GSE.index = df_dscovr_pos_GSE.index.tz_localize(None)

    print("Loading the L1 positions...")
    [l1_data, _] = pickle.load(open(data_dir / "l1_pos_GSE_19941113_20250731.p", "rb"))
    df_l1_pos_GSE = pd.DataFrame(l1_data)
    df_l1_pos_GSE.set_index("time", inplace=True)
    df_l1_pos_GSE.index.name = None
    df_l1_pos_GSE.index = df_l1_pos_GSE.index.tz_localize(None)

    # set the time that DSCOVR became the operational RTSW spacecraft according to swpc
    dscovr_start_time = datetime.datetime(2016, 7, 27, 16, 0, 0, tzinfo=None)
    end_time = datetime.datetime(2025, 7, 31, tzinfo=None)

    print(
        "Plotting the positions of Wind, ACE and DSCOVR in GSE coordinates since the start of DSCOVR..."
    )

    # plot the positions of Wind, ACE and DSCOVR in GSE coordinates since the start of DSCOVR
    df_dscovr_pos_GSE_since_dscovr = df_dscovr_pos_GSE[
        (df_dscovr_pos_GSE.index >= dscovr_start_time)
        & (df_dscovr_pos_GSE.index <= end_time)
    ]
    df_ace_pos_GSE_since_dscovr = df_ace_pos_GSE[
        (df_ace_pos_GSE.index >= dscovr_start_time) & (df_ace_pos_GSE.index <= end_time)
    ]
    df_wind_pos_GSE_since_dscovr = df_wind_pos_GSE[
        (df_wind_pos_GSE.index >= dscovr_start_time)
        & (df_wind_pos_GSE.index <= end_time)
    ]
    df_l1_pos_GSE_since_dscovr = df_l1_pos_GSE[
        (df_l1_pos_GSE.index >= dscovr_start_time) & (df_l1_pos_GSE.index <= end_time)
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        df_dscovr_pos_GSE_since_dscovr["x"],
        df_dscovr_pos_GSE_since_dscovr["y"],
        label="DSCOVR",
        color=geo_cornflowerblue,
    )
    ax.plot(
        df_ace_pos_GSE_since_dscovr["x"],
        df_ace_pos_GSE_since_dscovr["y"],
        label="ACE",
        color=geo_lime,
    )
    ax.plot(
        df_wind_pos_GSE_since_dscovr["x"],
        df_wind_pos_GSE_since_dscovr["y"],
        label="Wind",
        color=geo_magenta,
    )
    ax.plot(
        df_l1_pos_GSE_since_dscovr["x"],
        df_l1_pos_GSE_since_dscovr["y"],
        label="L1",
        color="black",
    )

    ax.set_xlabel("X (GSE) [km]")
    ax.set_ylabel("Y (GSE) [km]")

    # Force scientific notation with exponent 6
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((6, 6))  # force 1e6 notation
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend()
    plt.grid()
    plt.savefig(plot_dir / "positions_dscovr_ace_wind.pdf", dpi=300)
    plt.close()

    # Plot for ML Helio

    fig_ml, ax_ml = plt.subplots(figsize=(6, 6))

    # plot positions with inverted x-axis
    ax_ml.plot(
        -df_dscovr_pos_GSE_since_dscovr["x"],
        df_dscovr_pos_GSE_since_dscovr["y"],
        label="DSCOVR",
        color=geo_cornflowerblue,
    )
    ax_ml.plot(
        -df_ace_pos_GSE_since_dscovr["x"],
        df_ace_pos_GSE_since_dscovr["y"],
        label="ACE",
        color=geo_lime,
    )
    ax_ml.plot(
        -df_wind_pos_GSE_since_dscovr["x"],
        df_wind_pos_GSE_since_dscovr["y"],
        label="Wind",
        color=geo_magenta,
    )
    ax_ml.plot(
        -df_l1_pos_GSE_since_dscovr["x"],
        df_l1_pos_GSE_since_dscovr["y"],
        label="L1",
        color="black",
    )

    ax_ml.set_xlabel("X (GSE) [km]")
    ax_ml.set_ylabel("Y (GSE) [km]")

    # Force scientific notation with exponent 6
    formatter_ml = ScalarFormatter(useMathText=True)
    formatter_ml.set_scientific(True)
    formatter_ml.set_powerlimits((6, 6))  # force 1e6 notation
    ax_ml.yaxis.set_major_formatter(formatter_ml)
    ax_ml.xaxis.set_major_formatter(formatter_ml)

    ax_ml.legend()
    plt.savefig(plot_dir / "positions_dscovr_ace_wind_mlhelio.pdf", dpi=300)
    plt.close()

    print(
        "Calculating the differences and analyzing precise timeshift for each event..."
    )

    # Create a DataFrame to store the positions
    pos_df = pd.DataFrame(
        np.nan,
        index=df_wind_pos_GSE_since_dscovr.index,
        columns=[
            "wind_x",
            "wind_y",
            "wind_z",
            "ace_x",
            "ace_y",
            "ace_z",
            "dscovr_x",
            "dscovr_y",
            "dscovr_z",
        ],
    )
    # Fill the DataFrame with the positions
    pos_df["wind_x"] = df_wind_pos_GSE_since_dscovr["x"]
    pos_df["wind_y"] = df_wind_pos_GSE_since_dscovr["y"]
    pos_df["wind_z"] = df_wind_pos_GSE_since_dscovr["z"]

    pos_df["ace_x"] = df_ace_pos_GSE_since_dscovr["x"]
    pos_df["ace_y"] = df_ace_pos_GSE_since_dscovr["y"]
    pos_df["ace_z"] = df_ace_pos_GSE_since_dscovr["z"]

    pos_df["dscovr_x"] = df_dscovr_pos_GSE_since_dscovr["x"]
    pos_df["dscovr_y"] = df_dscovr_pos_GSE_since_dscovr["y"]
    pos_df["dscovr_z"] = df_dscovr_pos_GSE_since_dscovr["z"]

    # resample to 1 hour
    pos_df = pos_df.resample("1H").mean()

    # calculate the absolute distances between the spacecraft
    pos_df["distance_rtsw_wind"] = np.sqrt(
        (pos_df["wind_x"] - pos_df["dscovr_x"]) ** 2
        + (pos_df["wind_y"] - pos_df["dscovr_y"]) ** 2
        + (pos_df["wind_z"] - pos_df["dscovr_z"]) ** 2
    )

    pos_df["distance_ace_wind"] = np.sqrt(
        (pos_df["wind_x"] - pos_df["ace_x"]) ** 2
        + (pos_df["wind_y"] - pos_df["ace_y"]) ** 2
        + (pos_df["wind_z"] - pos_df["ace_z"]) ** 2
    )

    pos_df["distance_dscovr_ace"] = np.sqrt(
        (pos_df["dscovr_x"] - pos_df["ace_x"]) ** 2
        + (pos_df["dscovr_y"] - pos_df["ace_y"]) ** 2
        + (pos_df["dscovr_z"] - pos_df["ace_z"]) ** 2
    )

    pos_df["radial_distance_rtsw_wind"] = np.abs(pos_df["wind_x"] - pos_df["dscovr_x"])
    pos_df["radial_distance_ace_wind"] = np.abs(pos_df["wind_x"] - pos_df["ace_x"])
    pos_df["radial_distance_dscovr_ace"] = np.abs(pos_df["dscovr_x"] - pos_df["ace_x"])

    max_v_sw = 550  # km/s, maximum solar wind speed
    min_v_sw = 350  # km/s, minimum solar wind speed

    # calculate the time in minutes it takes to go from one spacecraft to the other based on max_v_sw
    pos_df["max_time_rtsw_wind"] = pos_df["distance_rtsw_wind"] / (
        max_v_sw * 60
    )  # in minutes
    pos_df["max_time_ace_wind"] = pos_df["distance_ace_wind"] / (
        max_v_sw * 60
    )  # in minutes
    pos_df["max_time_dscovr_ace"] = pos_df["distance_dscovr_ace"] / (
        max_v_sw * 60
    )  # in minutes

    # calculate the time in minutes it takes to go from one spacecraft to the other based on min_v_sw
    pos_df["min_time_rtsw_wind"] = pos_df["distance_rtsw_wind"] / (
        min_v_sw * 60
    )  # in minutes
    pos_df["min_time_ace_wind"] = pos_df["distance_ace_wind"] / (
        min_v_sw * 60
    )  # in minutes
    pos_df["min_time_dscovr_ace"] = pos_df["distance_dscovr_ace"] / (
        min_v_sw * 60
    )  # in minutes

    # calculate the time in minutes it takes to go from one spacecraft to the other based on max_v_sw for radial distances
    pos_df["max_time_radial_rtsw_wind"] = pos_df["radial_distance_rtsw_wind"] / (
        max_v_sw * 60
    )  # in minutes
    pos_df["max_time_radial_ace_wind"] = pos_df["radial_distance_ace_wind"] / (
        max_v_sw * 60
    )  # in minutes
    pos_df["max_time_radial_dscovr_ace"] = pos_df["radial_distance_dscovr_ace"] / (
        max_v_sw * 60
    )  # in minutes

    # calculate the time in minutes it takes to go from one spacecraft to the other based on min_v_sw for radial distances
    pos_df["min_time_radial_rtsw_wind"] = pos_df["radial_distance_rtsw_wind"] / (
        min_v_sw * 60
    )  # in minutes
    pos_df["min_time_radial_ace_wind"] = pos_df["radial_distance_ace_wind"] / (
        min_v_sw * 60
    )  # in minutes
    pos_df["min_time_radial_dscovr_ace"] = pos_df["radial_distance_dscovr_ace"] / (
        min_v_sw * 60
    )  # in minutes

    print("Saving the results to a CSV file...")
    pos_df.to_csv(data_dir / "omni_shift_analysis_results_gse.csv")

    # print statistics about the distances
    print(
        f"Mean absolute distance between DSCOVR and Wind: {pos_df['distance_rtsw_wind'].abs().mean():.2f} km"
    )
    print(
        f"Max absolute distance between DSCOVR and Wind: {pos_df['distance_rtsw_wind'].abs().max():.2f} km"
    )
    print(
        f"Min absolute distance between DSCOVR and Wind: {pos_df['distance_rtsw_wind'].abs().min():.2f} km"
    )
    print(
        f"Mean absolute distance between ACE and Wind: {pos_df['distance_ace_wind'].abs().mean():.2f} km"
    )
    print(
        f"Max absolute distance between ACE and Wind: {pos_df['distance_ace_wind'].abs().max():.2f} km"
    )
    print(
        f"Min absolute distance between ACE and Wind: {pos_df['distance_ace_wind'].abs().min():.2f} km"
    )
    print(
        f"Mean absolute distance between DSCOVR and ACE: {pos_df['distance_dscovr_ace'].abs().mean():.2f} km"
    )
    print(
        f"Max absolute distance between DSCOVR and ACE: {pos_df['distance_dscovr_ace'].abs().max():.2f} km"
    )
    print(
        f"Min absolute distance between DSCOVR and ACE: {pos_df['distance_dscovr_ace'].abs().min():.2f} km"
    )

    print(
        "########################################################################################"
    )

    # print statistics about the radial distances
    print(
        f"Mean radial distance between DSCOVR and Wind: {pos_df['radial_distance_rtsw_wind'].abs().mean():.2f} km"
    )
    print(
        f"Max radial distance between DSCOVR and Wind: {pos_df['radial_distance_rtsw_wind'].abs().max():.2f} km"
    )
    print(
        f"Min radial distance between DSCOVR and Wind: {pos_df['radial_distance_rtsw_wind'].abs().min():.2f} km"
    )
    print(
        f"Mean radial distance between ACE and Wind: {pos_df['radial_distance_ace_wind'].abs().mean():.2f} km"
    )
    print(
        f"Max radial distance between ACE and Wind: {pos_df['radial_distance_ace_wind'].abs().max():.2f} km"
    )
    print(
        f"Min radial distance between ACE and Wind: {pos_df['radial_distance_ace_wind'].abs().min():.2f} km"
    )
    print(
        f"Mean radial distance between DSCOVR and ACE: {pos_df['radial_distance_dscovr_ace'].abs().mean():.2f} km"
    )
    print(
        f"Max radial distance between DSCOVR and ACE: {pos_df['radial_distance_dscovr_ace'].abs().max():.2f} km"
    )
    print(
        f"Min radial distance between DSCOVR and ACE: {pos_df['radial_distance_dscovr_ace'].abs().min():.2f} km"
    )

    print(
        "########################################################################################"
    )

    # print statistics about the times
    print(
        f"Mean time to go from DSCOVR to Wind at max solar wind speed: {pos_df['max_time_rtsw_wind'].mean():.2f} minutes"
    )
    print(
        f"Max time to go from DSCOVR to Wind at max solar wind speed: {pos_df['max_time_rtsw_wind'].max():.2f} minutes"
    )
    print(
        f"Min time to go from DSCOVR to Wind at max solar wind speed: {pos_df['max_time_rtsw_wind'].min():.2f} minutes"
    )
    print(
        f"Mean time to go from ACE to Wind at max solar wind speed: {pos_df['max_time_ace_wind'].mean():.2f} minutes"
    )
    print(
        f"Max time to go from ACE to Wind at max solar wind speed: {pos_df['max_time_ace_wind'].max():.2f} minutes"
    )
    print(
        f"Min time to go from ACE to Wind at max solar wind speed: {pos_df['max_time_ace_wind'].min():.2f} minutes"
    )
    print(
        f"Mean time to go from DSCOVR to ACE at max solar wind speed: {pos_df['max_time_dscovr_ace'].mean():.2f} minutes"
    )
    print(
        f"Max time to go from DSCOVR to ACE at max solar wind speed: {pos_df['max_time_dscovr_ace'].max():.2f} minutes"
    )
    print(
        f"Min time to go from DSCOVR to ACE at max solar wind speed: {pos_df['max_time_dscovr_ace'].min():.2f} minutes"
    )

    print(
        "########################################################################################"
    )

    print(
        f"Mean time to go from DSCOVR to Wind at min solar wind speed: {pos_df['min_time_rtsw_wind'].mean():.2f} minutes"
    )
    print(
        f"Max time to go from DSCOVR to Wind at min solar wind speed: {pos_df['min_time_rtsw_wind'].max():.2f} minutes"
    )
    print(
        f"Min time to go from DSCOVR to Wind at min solar wind speed: {pos_df['min_time_rtsw_wind'].min():.2f} minutes"
    )
    print(
        f"Mean time to go from ACE to Wind at min solar wind speed: {pos_df['min_time_ace_wind'].mean():.2f} minutes"
    )
    print(
        f"Max time to go from ACE to Wind at min solar wind speed: {pos_df['min_time_ace_wind'].max():.2f} minutes"
    )
    print(
        f"Min time to go from ACE to Wind at min solar wind speed: {pos_df['min_time_ace_wind'].min():.2f} minutes"
    )
    print(
        f"Mean time to go from DSCOVR to ACE at min solar wind speed: {pos_df['min_time_dscovr_ace'].mean():.2f} minutes"
    )
    print(
        f"Max time to go from DSCOVR to ACE at min solar wind speed: {pos_df['min_time_dscovr_ace'].max():.2f} minutes"
    )
    print(
        f"Min time to go from DSCOVR to ACE at min solar wind speed: {pos_df['min_time_dscovr_ace'].min():.2f} minutes"
    )

    print(
        "########################################################################################"
    )

    print(
        "Mean time to go from DSCOVR to Wind at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_rtsw_wind'].mean():.2f} minutes"
    )
    print(
        "Max time to go from DSCOVR to Wind at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_rtsw_wind'].max():.2f} minutes"
    )
    print(
        "Min time to go from DSCOVR to Wind at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_rtsw_wind'].min():.2f} minutes"
    )
    print(
        "Mean time to go from ACE to Wind at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_ace_wind'].mean():.2f} minutes"
    )
    print(
        "Max time to go from ACE to Wind at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_ace_wind'].max():.2f} minutes"
    )
    print(
        "Min time to go from ACE to Wind at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_ace_wind'].min():.2f} minutes"
    )
    print(
        "Mean time to go from DSCOVR to ACE at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_dscovr_ace'].mean():.2f} minutes"
    )
    print(
        "Max time to go from DSCOVR to ACE at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_dscovr_ace'].max():.2f} minutes"
    )
    print(
        "Min time to go from DSCOVR to ACE at max solar wind speed for radial distances: "
        f"{pos_df['max_time_radial_dscovr_ace'].min():.2f} minutes"
    )

    print(
        "########################################################################################"
    )

    print(
        "Mean time to go from DSCOVR to Wind at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_rtsw_wind'].mean():.2f} minutes"
    )
    print(
        "Max time to go from DSCOVR to Wind at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_rtsw_wind'].max():.2f} minutes"
    )
    print(
        "Min time to go from DSCOVR to Wind at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_rtsw_wind'].min():.2f} minutes"
    )
    print(
        "Mean time to go from ACE to Wind at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_ace_wind'].mean():.2f} minutes"
    )
    print(
        "Max time to go from ACE to Wind at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_ace_wind'].max():.2f} minutes"
    )
    print(
        "Min time to go from ACE to Wind at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_ace_wind'].min():.2f} minutes"
    )
    print(
        "Mean time to go from DSCOVR to ACE at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_dscovr_ace'].mean():.2f} minutes"
    )
    print(
        "Max time to go from DSCOVR to ACE at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_dscovr_ace'].max():.2f} minutes"
    )
    print(
        "Min time to go from DSCOVR to ACE at min solar wind speed for radial distances: "
        f"{pos_df['min_time_radial_dscovr_ace'].min():.2f} minutes"
    )

    print(
        "########################################################################################"
    )

    print("Loading shocks database from Helsinki...")
    shocks_file = data_dir / "shocks_20250814_160151.dat"
    df_shocks = pd.read_csv(
        shocks_file,
        delimiter=";",
        header=None,
        names=["year", "month", "day", "hour", "minute", "second", "spacecraft"],
    )
    df_shocks["time"] = pd.to_datetime(
        df_shocks[["year", "month", "day", "hour", "minute", "second"]]
    )

    wind_shocks = df_shocks[df_shocks["spacecraft"].str.contains("Wind")]
    ace_shocks = df_shocks[df_shocks["spacecraft"].str.contains("ACE")]
    dscovr_shocks = df_shocks[df_shocks["spacecraft"].str.contains("DSCOVR")]

    associated_shocks_df = pd.DataFrame(
        pd.NaT, index=np.arange(len(wind_shocks)), columns=["Wind", "ACE", "DSCOVR"]
    )

    associated_shocks_df["Wind"] = wind_shocks["time"].values

    # for every ACE shock, find the closest Wind shock
    for i, ace_shock in ace_shocks.iterrows():
        ace_time = ace_shock["time"]
        # find the closest Wind shock
        wind_nearest_time = associated_shocks_df["Wind"].sub(ace_time).abs().idxmin()
        associated_shocks_df.loc[wind_nearest_time, "ACE"] = ace_time

    # for every DSCOVR shock, find the closest Wind shock
    for i, dscovr_shock in dscovr_shocks.iterrows():
        dscovr_time = dscovr_shock["time"]
        # find the closest Wind shock
        wind_nearest_time = associated_shocks_df["Wind"].sub(dscovr_time).abs().idxmin()
        associated_shocks_df.loc[wind_nearest_time, "DSCOVR"] = dscovr_time

    associated_shocks_df["Wind_ACE_diff"] = (
        associated_shocks_df["Wind"] - associated_shocks_df["ACE"]
    ).abs().dt.total_seconds() / 60  # in minutes
    associated_shocks_df["Wind_DSCOVR_diff"] = (
        associated_shocks_df["Wind"] - associated_shocks_df["DSCOVR"]
    ).abs().dt.total_seconds() / 60  # in minutes
    associated_shocks_df["ACE_DSCOVR_diff"] = (
        associated_shocks_df["ACE"] - associated_shocks_df["DSCOVR"]
    ).abs().dt.total_seconds() / 60  # in minutes

    max_ass_diff = 40

    # number of non Nan values in each difference column
    print(
        f"Number of Wind-ACE differences: {associated_shocks_df['Wind_ACE_diff'].notna().sum()}"
    )
    print(
        f"Number of differences below {max_ass_diff} minutes: {associated_shocks_df['Wind_ACE_diff'][associated_shocks_df['Wind_ACE_diff'] < max_ass_diff].count()}"
    )
    print(
        f"Number of Wind-DSCOVR differences: {associated_shocks_df['Wind_DSCOVR_diff'].notna().sum()}"
    )
    print(
        f"Number of differences below {max_ass_diff} minutes: {associated_shocks_df['Wind_DSCOVR_diff'][associated_shocks_df['Wind_DSCOVR_diff'] < max_ass_diff].count()}"
    )
    print(
        f"Number of ACE-DSCOVR differences: {associated_shocks_df['ACE_DSCOVR_diff'].notna().sum()}"
    )
    print(
        f"Number of differences below {max_ass_diff} minutes: {associated_shocks_df['ACE_DSCOVR_diff'][associated_shocks_df['ACE_DSCOVR_diff'] < max_ass_diff].count()}"
    )

    print(
        "#########################################################################################"
    )

    print(
        f"Mean Wind-ACE difference below {max_ass_diff} minutes: {associated_shocks_df['Wind_ACE_diff'][associated_shocks_df['Wind_ACE_diff'] < max_ass_diff].mean():.2f} minutes"
    )
    print(
        f"Mean Wind-DSCOVR difference below {max_ass_diff} minutes: {associated_shocks_df['Wind_DSCOVR_diff'][associated_shocks_df['Wind_DSCOVR_diff'] < max_ass_diff].mean():.2f} minutes"
    )
    print(
        f"Mean ACE-DSCOVR difference below {max_ass_diff} minutes: {associated_shocks_df['ACE_DSCOVR_diff'][associated_shocks_df['ACE_DSCOVR_diff'] < max_ass_diff].mean():.2f} minutes"
    )

    # # plot the differences if they are below 60 minutes
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.scatter(range(associated_shocks_df['ACE_DSCOVR_diff'][associated_shocks_df['ACE_DSCOVR_diff'] < 60].count()),
    #            associated_shocks_df['ACE_DSCOVR_diff'][associated_shocks_df['ACE_DSCOVR_diff'] < 60],
    #            label="ACE-DSCOVR", color=geo_lime, s=10)
    # ax.scatter(range(associated_shocks_df['Wind_ACE_diff'][associated_shocks_df['Wind_ACE_diff'] < 60].count()),
    #            associated_shocks_df['Wind_ACE_diff'][associated_shocks_df['Wind_ACE_diff'] < 60],
    #            label="Wind-ACE", color=geo_cornflowerblue, s=10)
    # ax.scatter(range(associated_shocks_df['Wind_DSCOVR_diff'][associated_shocks_df['Wind_DSCOVR_diff'] < 60].count()),
    #            associated_shocks_df['Wind_DSCOVR_diff'][associated_shocks_df['Wind_DSCOVR_diff'] < 60],
    #            label="Wind-DSCOVR", color=geo_magenta, s=10)

    # ax.set_xlabel("Shock Index")
    # ax.set_ylabel("Time Difference [minutes]")
    # ax.legend()
    # plt.grid()
    # plt.show()

    if Path(data_dir / "event_wise_shift.csv").exists():
        print("Event-wise shift data already exists. Loading existing data...")
        results_df = pd.read_csv(data_dir / "event_wise_shift.csv", index_col=0)
    else:
        print("No event-wise shift data found. Creating new DataFrame...")

        print("Loading the in situ rtsw data...")
        # Load in situ rtsw data for getting the rtsw positions
        insitu_data_path = Path(data_dir / "noaa_archive_gsm.p")

        [data, _] = pickle.load(open(insitu_data_path, "rb"))

        dataframe = pd.DataFrame(data)
        dataframe.set_index("time", inplace=True)
        dataframe.index.name = None
        dataframe.index = dataframe.index.tz_localize(None)

        nr_wind_events = len(iind)
        print(f"Number of Wind events: {nr_wind_events}")

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(
            np.nan,
            index=range(nr_wind_events),
            columns=[
                "icme_start_time",
                "mo_start_time",
                "mo_end_time",
                "wind_x",
                "wind_y",
                "wind_z",
                "rtsw_x",
                "rtsw_y",
                "rtsw_z",
                "icme_speed_mean",
            ],
        )

        for event_nr in range(nr_wind_events):
            print(f"Processing event {event_nr + 1}/{nr_wind_events}")
            i = iind[event_nr]
            icme_start_time = pd.to_datetime(
                icmecat.loc[i, "icme_start_time"]
            ).tz_localize(None)
            mo_start_time = pd.to_datetime(icmecat.loc[i, "mo_start_time"]).tz_localize(
                None
            )
            mo_end_time = pd.to_datetime(icmecat.loc[i, "mo_end_time"]).tz_localize(
                None
            )
            icme_speed_mean = icmecat.loc[event_nr, "icme_speed_mean"]

            # whenever icme_speed_mean is NaN, use mo_speed_mean instead
            if np.isnan(icme_speed_mean):
                icme_speed_mean = icmecat.loc[event_nr, "mo_speed_mean"]

            # fill the results DataFrame
            results_df.loc[event_nr, "icme_start_time"] = icme_start_time
            results_df.loc[event_nr, "mo_start_time"] = mo_start_time
            results_df.loc[event_nr, "mo_end_time"] = mo_end_time
            results_df.loc[event_nr, "icme_speed_mean"] = icme_speed_mean

            rtsw_nearest_time = dataframe.index.get_indexer(
                [icme_start_time], method="nearest", tolerance=pd.Timedelta("1h")
            )

            if rtsw_nearest_time[0] == -1:
                print(f"No close timestamp found for {icme_start_time} in RTSW data")
                results_df.loc[event_nr, "rtsw_x"] = np.nan
                results_df.loc[event_nr, "rtsw_y"] = np.nan
                results_df.loc[event_nr, "rtsw_z"] = np.nan
            else:
                # check the source at that time
                source_mag = dataframe.iloc[rtsw_nearest_time[0]]["source_mag"]

                if source_mag == 1.0:
                    dscovr_nearest_time = df_dscovr_pos_GSE.index.get_indexer(
                        [icme_start_time],
                        method="nearest",
                        tolerance=pd.Timedelta("1h"),
                    )
                    if dscovr_nearest_time[0] == -1:
                        print(
                            f"No close timestamp found for {icme_start_time} in DSCOVR data"
                        )
                        results_df.loc[event_nr, "rtsw_x"] = np.nan
                        results_df.loc[event_nr, "rtsw_y"] = np.nan
                        results_df.loc[event_nr, "rtsw_z"] = np.nan
                    else:
                        dscovr_row = df_dscovr_pos_GSE.iloc[dscovr_nearest_time[0]]
                        results_df.loc[event_nr, "rtsw_x"] = dscovr_row["x"]
                        results_df.loc[event_nr, "rtsw_y"] = dscovr_row["y"]
                        results_df.loc[event_nr, "rtsw_z"] = dscovr_row["z"]

                elif source_mag == 2.0:
                    ace_nearest_time = df_ace_pos_GSE.index.get_indexer(
                        [icme_start_time],
                        method="nearest",
                        tolerance=pd.Timedelta("1h"),
                    )
                    if ace_nearest_time[0] == -1:
                        print(
                            f"No close timestamp found for {icme_start_time} in ACE data"
                        )
                        results_df.loc[event_nr, "rtsw_x"] = np.nan
                        results_df.loc[event_nr, "rtsw_y"] = np.nan
                        results_df.loc[event_nr, "rtsw_z"] = np.nan
                    else:
                        ace_row = df_ace_pos_GSE.iloc[ace_nearest_time[0]]
                        results_df.loc[event_nr, "rtsw_x"] = ace_row["x"]
                        results_df.loc[event_nr, "rtsw_y"] = ace_row["y"]
                        results_df.loc[event_nr, "rtsw_z"] = ace_row["z"]

            wind_nearest_time = df_wind_pos_GSE.index.get_indexer(
                [icme_start_time], method="nearest", tolerance=pd.Timedelta("1h")
            )
            if wind_nearest_time[0] == -1:
                print(f"No close timestamp found for {icme_start_time} in Wind data")
                results_df.loc[event_nr, "wind_x"] = np.nan
                results_df.loc[event_nr, "wind_y"] = np.nan
                results_df.loc[event_nr, "wind_z"] = np.nan
            else:
                wind_row = df_wind_pos_GSE.iloc[wind_nearest_time[0]]
                results_df.loc[event_nr, "wind_x"] = wind_row["x"]
                results_df.loc[event_nr, "wind_y"] = wind_row["y"]
                results_df.loc[event_nr, "wind_z"] = wind_row["z"]

        # save the results_df to a file
        results_df.to_csv(data_dir / "event_wise_shift.csv")

    results_df = results_df.dropna()
    results_df["radial_distance"] = np.abs(results_df["rtsw_x"] - results_df["wind_x"])
    results_df["time_radial"] = results_df["radial_distance"] / (
        results_df["icme_speed_mean"] * 60
    )  # in minutes

    print("Maximum radial distance: ", results_df["radial_distance"].max())
    print("Mean radial distance: ", results_df["radial_distance"].mean())
    print("Minimum radial distance: ", results_df["radial_distance"].min())

    print("Maximum time radial: ", results_df["time_radial"].max())
    print("Mean time radial: ", results_df["time_radial"].mean())
    print("Minimum time radial: ", results_df["time_radial"].min())
