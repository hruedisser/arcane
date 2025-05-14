import glob
from datetime import timedelta

import pandas as pd
from scipy.io import netcdf


def get_dscovrpos(fp):
    """raw = gse"""
    try:
        ncdf = netcdf.NetCDFFile(fp, "r")
        data = {
            df_col: ncdf.variables[cdf_col][:]
            for cdf_col, df_col in zip(
                ["time", "sat_x_gse", "sat_y_gse", "sat_z_gse"], ["time", "x", "y", "z"]
            )
        }
        df = pd.DataFrame.from_dict(data)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
    except Exception as e:
        print("ERROR:", e, fp)
        df = None
    return df


def get_dscovrpositions(start_timestamp, end_timestamp, kernels_path):
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        print(start)
        year = start.year
        date_str = f"{year}{start.month:02}{start.day:02}"
        fn = glob.glob(f"{kernels_path}" + f"/oe_pop_dscovr_s{date_str}000000_*.nc")
        if fn:
            _df = get_dscovrpos(fn[0])
            if _df is not None:
                if df is None:
                    df = _df.copy(deep=True)
                else:
                    df = pd.concat([df, _df])
        start += timedelta(days=1)
    df = df.reset_index(drop=True)
    return df
