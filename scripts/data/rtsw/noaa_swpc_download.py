import datetime
import gzip
import json
import logging
import os
import pickle
import shutil
import urllib
import urllib.error
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from . import functions_noaa as fa
from . import position_frame_transforms as pos_transform

####################################################################################
##############################                        ##############################
##############################      INITIALIZING      ##############################
##############################                        ##############################
####################################################################################

# Initialize variables
base_url = "http://services.swpc.noaa.gov/text/rtsw/data/"
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, "../../.."))

out = os.path.join(base_dir, "data")
kernels_path = os.path.join(out, "kernels")

print(f"Base directory: {base_dir}")
print(f"Data directory: {out}")
print(f"Kernels directory: {kernels_path}")

error_count = 0
stopflag = False
counter = 0
types = ["mag", "plasma", "kp"]
timespan = "1-day"

download_kernels = False
download_data = False
process = False
create_big = True


start_timestamp = datetime.datetime(2016, 7, 26)
# today minus 1 day
end_timestamp = datetime.datetime.now() - datetime.timedelta(days=1)

####################################################################################
##############################                        ##############################
##############################       FUNCTIONS        ##############################
##############################                        ##############################
####################################################################################


def convert_istp_time(istp_time):
    """Convert ISTP time format to datetime object."""
    year = int(str(istp_time[0])[:4])
    day_of_year = int(str(istp_time[0])[4:])
    milliseconds = istp_time[1]
    date = (
        datetime.datetime(year, 1, 1)
        + datetime.timedelta(day_of_year - 1)
        + datetime.timedelta(milliseconds=milliseconds)
    )
    return date


def get_noaa_data(json_file):
    data = defaultdict(dict)
    with open(json_file, "r") as jdata:
        dp = json.load(jdata)
        columns = dp[0]  # Extract column names
        for entry in dp[1:]:  # Skip the first entry which contains column names
            entry_dict = dict(
                zip(columns, entry)
            )  # Map each entry to its corresponding column name
            time_tag = entry_dict.get("time_tag")
            active = int(entry_dict.get("active"))
            if time_tag is not None and active is not None:
                for key, value in entry_dict.items():
                    if key != "time_tag":
                        entry_dict[key] = float(value) if value is not None else np.nan
                # print(active)
                if (
                    time_tag not in data and active == 1
                ):  # or float(source) > float(data[time_tag].get('source', float('-inf'))):
                    data[time_tag] = entry_dict

    return list(data.values())


def to_epoch_millis_utc(dt):
    dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt_utc.timestamp() * 1000)


def extract_wget_links(start_date_str, end_date_str):
    start_dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    # Add one day, subtract one millisecond to reach 23:59:59.999
    end_dt = datetime.datetime.strptime(end_date_str, "%Y-%m-%d") + datetime.timedelta(
        days=1, milliseconds=-1
    )

    start_ms = to_epoch_millis_utc(start_dt)
    end_ms = to_epoch_millis_utc(end_dt)

    api_url = f"https://www.ngdc.noaa.gov/dscovr-data-access/files?start_date={start_ms}&end_date={end_ms}"

    print(f"Requesting file list from: {api_url}")

    response = requests.get(api_url)

    if response.status_code != 200:
        print(f"Error: Unable to fetch data from {api_url}")
        return []

    file_list = response.json()

    # Collect only 'pop' URLs
    file_urls = []
    for date_key, file_types in file_list.items():
        if isinstance(file_types, dict) and "pop" in file_types:
            file_urls.append(file_types["pop"])

    return file_urls


###################################################################################
##############################                       ##############################
##############################    DOWNLOAD POSITIONS      ##############################
##############################                       ##############################
###################################################################################

if download_kernels == True:

    # check if kernels_path exists
    if not os.path.exists(kernels_path):
        print(f"Warning: {kernels_path} does not exist.")

    # check if out exists
    if not os.path.exists(out):
        print(f"Warning: {out} does not exist.")

    print(f"Downloading DSCOVR kernels from {start_timestamp} to {end_timestamp}")

    # Get the wget links
    file_urls = extract_wget_links(
        start_timestamp.strftime("%Y-%m-%d"), end_timestamp.strftime("%Y-%m-%d")
    )

    # Save the links to a file
    with open(os.path.join(out, "wget_links.txt"), "w") as f:
        for link in file_urls:
            f.write(link + "\n")

    print(
        f"Extracted {len(file_urls)} wget links and saved to {os.path.join(out, 'wget_links.txt')}"
    )

    # Download the files and extract them
    for file_url in file_urls:
        filename = os.path.basename(file_url)
        file_path = os.path.join(kernels_path, filename)

        print(f"Downloading {file_url} to {file_path}")
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
            continue
        # Extract the file if it's a gzip file
        if filename.endswith(".gz"):
            with open(file_path[:-3], "wb") as f_out:
                with gzip.open(file_path, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            print(f"Extracted {filename[:-3]} from {filename}")


###################################################################################
##############################                       ##############################
##############################    GET POSITIONS      ##############################
##############################                       ##############################
###################################################################################

if create_big == True:

    # get ACE positions from url

    # Load the data
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

    # get DSCOVR positions from kernels
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

####################################################################################
##############################                        ##############################
##############################    DOWNLOADING DATA    ##############################
##############################                        ##############################
####################################################################################

if download_data == True:
    while stopflag == False:
        try:
            for filetyp in types:
                print(f"{filetyp}-{timespan}.{counter}.json")
                # sys.stdout.flush()
                urllib.request.urlretrieve(
                    base_url + filetyp + "-" + timespan + "." + str(counter) + ".json",
                    out + "/" + filetyp + timespan + "." + str(counter) + ".json",
                )
        except urllib.error.URLError as e:
            logging.error(e.reason)
            error_count += 1
            # if error_count >= 4:
            #    stopflag = True
            print(type(e))
        else:
            # Reset error count if successful request
            error_count = 0
        finally:
            counter = counter + 1
            if counter >= 100000:
                stopflag = True


###################################################################################
##############################                       ##############################
##############################    PROCESSING DATA    ##############################
##############################                       ##############################
###################################################################################

if process == True:

    # Initialize empty lists to store combined data
    mega_mag_data = []
    mega_plasma_data = []

    for i in range(12000):
        try:
            print(i)
            magname = out + "/mag1-day." + str(i) + ".json"
            plasmaname = out + "/plasma1-day." + str(i) + ".json"

            # Append data to lists
            mag_data = get_noaa_data(magname)
            plasma_data = get_noaa_data(plasmaname)

            mega_mag_data.extend(mag_data)
            mega_plasma_data.extend(plasma_data)
        except:
            continue

    # Sort the combined data by 'time_tag'
    mega_mag_data.sort(key=lambda x: x["time_tag"])
    mega_plasma_data.sort(key=lambda x: x["time_tag"])

    # Convert to recarray
    dtype_mag = np.dtype(
        [
            (key, "O") if key == "time_tag" else (key, "f")
            for key in mega_mag_data[0].keys()
        ]
    )
    recarray_mag = np.rec.fromrecords(
        [tuple(entry.values()) for entry in mega_mag_data], dtype=dtype_mag
    )

    dtype_plasma = np.dtype(
        [
            (key, "O") if key == "time_tag" else (key, "f")
            for key in mega_plasma_data[0].keys()
        ]
    )
    recarray_plasma = np.rec.fromrecords(
        [tuple(entry.values()) for entry in mega_plasma_data], dtype=dtype_plasma
    )

    # Save as pickle file
    with open(out + "/mega_data_mag.pkl", "wb") as f:
        pickle.dump(recarray_mag, f)

    # Save as pickle file
    with open(out + "/mega_data_plasma.pkl", "wb") as f:
        pickle.dump(recarray_plasma, f)


###################################################################################
##############################                       ##############################
##############################    COMBINING DATA     ##############################
##############################                       ##############################
###################################################################################


if create_big == True:
    # Load the pickle files
    with open(out + "/mega_data_mag.pkl", "rb") as f:
        mag_data = pickle.load(f)

    with open(out + "/mega_data_plasma.pkl", "rb") as f:
        plasma_data = pickle.load(f)

    # Convert recarrays to DataFrames
    mag_df = pd.DataFrame(mag_data)
    plasma_df = pd.DataFrame(plasma_data)

    # Merge the DataFrames on 'time_tag' field
    combined_df = pd.merge(
        mag_df, plasma_df, on="time_tag", how="outer", suffixes=("_mag", "_plasma")
    )

    # Rearrange columns
    combined_df = combined_df[
        [
            "time_tag",
            "bt",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "lat_gsm",
            "lon_gsm",
            "source_mag",
            "speed",
            "density",
            "temperature",
            "source_plasma",
        ]
    ]

    combined_df["time_tag"] = pd.to_datetime(combined_df["time_tag"])

    datetime_objects = [
        datetime.datetime.strptime(str(dt), "%Y-%m-%d %H:%M:%S")
        for dt in combined_df["time_tag"]
    ]

    # Preprocess position data frames
    # df_ace_pos.set_index('time', inplace=True)
    # df_dscovr_pos.set_index('time', inplace=True)

    # Make array
    noaa = np.zeros(
        len(combined_df),
        dtype=[
            ("time", object),
            ("bx", float),
            ("by", float),
            ("bz", float),
            ("bt", float),
            ("np", float),
            ("vt", float),
            ("tp", float),
            ("source_mag", float),
            ("source_plasma", float),
            ("x", float),
            ("y", float),
            ("z", float),
            ("r", float),
            ("lat", float),
            ("lon", float),
        ],
    )

    # Convert to recarray
    noaa = noaa.view(np.recarray)

    # Fill with data
    noaa.time = datetime_objects
    noaa.bx = combined_df["bx_gsm"].values
    noaa.by = combined_df["by_gsm"].values
    noaa.bz = combined_df["bz_gsm"].values
    noaa.bt = combined_df["bt"].values
    noaa.np = combined_df["density"].values
    noaa.vt = combined_df["speed"].values
    noaa.tp = combined_df["temperature"].values
    noaa.source_mag = combined_df["source_mag"].values
    noaa.source_plasma = combined_df["source_plasma"].values

    print(noaa.source_mag)
    print(type(noaa.source_mag))
    print(type(noaa.source_mag[0]))

    # check if noaa.source_mag contains 1.0
    if np.any(noaa.source_mag == 1.0):
        print("DSCOVR data 1.0 found")
    else:
        print("No DSCOVR data 1.0 found")
    # check if noaa.source_mag contains 2.0
    if np.any(noaa.source_mag == 2.0):
        print("DSCOVR data 2.0 found")
    else:
        print("No DSCOVR data 2.0 found")

    # Use vectorized lookup for positions
    def get_positions(time, source_mag):
        if source_mag in [1.0, 2.0]:
            return (
                df_dscovr_pos.loc[time] if time in df_dscovr_pos.index else [np.nan] * 6
            )
        else:
            return df_ace_pos.loc[time] if time in df_ace_pos.index else [np.nan] * 6

    positions = np.array(
        [
            get_positions(time, source)
            for time, source in zip(noaa.time, noaa.source_mag)
        ]
    )

    noaa.x = positions[:, 0]
    noaa.y = positions[:, 1]
    noaa.z = positions[:, 2]
    noaa.r = positions[:, 3]
    noaa.lat = positions[:, 4]
    noaa.lon = positions[:, 5]

    header = (
        "Real time solar wind magnetic field and plasma data from NOAA, "
        + "obtained daily from http://services.swpc.noaa.gov/text/rtsw/data/  "
        + "Timerange: "
        + noaa.time[0].strftime("%Y-%b-%d %H:%M")
        + " to "
        + noaa.time[-1].strftime("%Y-%b-%d %H:%M")
        + "The data are available in a numpy recarray, fields can be accessed by nf.time, nf.bx, nf.vt etc. "
        + "Total number of data points: "
        + str(noaa.size)
        + ". "
        + "Units are btxyz [nT, GSM], vt  [km s^-1], np[cm^-3], tp [K]. "
        + "File creation date: "
        + datetime.datetime.utcnow().strftime("%Y-%b-%d %H:%M")
        + " UTC"
    )

    pickle.dump([noaa, header], open(out + "/noaa_archive_gsm_updated.p", "wb"))
