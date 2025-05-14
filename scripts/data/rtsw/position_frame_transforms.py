import itertools

import numpy as np
import pandas as pd

"""
Functions by E.E. Davies
"""


def cart2sphere(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2) / 1.495978707e8
    theta = np.arctan2(z, np.sqrt(x**2 + y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y, x) * 360 / 2 / np.pi
    return (r, theta, phi)


# input datetime to return T1, T2 and T3 based on Hapgood 1992
# http://www.igpp.ucla.edu/public/vassilis/ESS261/Lecture03/Hapgood_sdarticle.pdf
def get_geocentric_transformation_matrices(time):
    # Format dates correctly, calculate MJD, T0, UT
    ts = pd.Timestamp(time)
    jd = ts.to_julian_date()
    mjd = float(int(jd - 2400000.5))  # Use modified Julian date
    T0 = (mjd - 51544.5) / 36525.0
    UT = ts.hour + ts.minute / 60.0 + ts.second / 3600.0  # Time in UT in hours

    # Define position of geomagnetic pole in GEO coordinates
    pgeo = np.deg2rad(
        78.8 + 4.283 * ((mjd - 46066) / 365.25) * 0.01
    )  # Convert to radians
    lgeo = np.deg2rad(
        289.1 - 1.413 * ((mjd - 46066) / 365.25) * 0.01
    )  # Convert to radians

    # GEO vector
    Qg = np.array(
        [np.cos(pgeo) * np.cos(lgeo), np.cos(pgeo) * np.sin(lgeo), np.sin(pgeo)]
    )

    # CREATE T1
    zeta = np.deg2rad(100.461 + 36000.770 * T0 + 15.04107 * UT)
    cos_z, sin_z = np.cos(zeta), np.sin(zeta)
    T1 = np.array([[cos_z, sin_z, 0], [-sin_z, cos_z, 0], [0, 0, 1]])

    # CREATE T2
    LAMBDA = 280.460 + 36000.772 * T0 + 0.04107 * UT
    M = 357.528 + 35999.050 * T0 + 0.04107 * UT
    M_rad = np.deg2rad(M)

    lt2 = np.deg2rad(
        LAMBDA + (1.915 - 0.0048 * T0) * np.sin(M_rad) + 0.020 * np.sin(2 * M_rad)
    )
    cos_lt2, sin_lt2 = np.cos(lt2), np.sin(lt2)
    t2z = np.array([[cos_lt2, sin_lt2, 0], [-sin_lt2, cos_lt2, 0], [0, 0, 1]])

    et2 = np.deg2rad(23.439 - 0.013 * T0)
    cos_e, sin_e = np.cos(et2), np.sin(et2)
    t2x = np.array([[1, 0, 0], [0, cos_e, sin_e], [0, -sin_e, cos_e]])

    T2 = t2z @ t2x  # Matrix multiplication

    # Compute Qe
    T2T1t = T2 @ T1.T
    Qe = T2T1t @ Qg
    psigsm = np.arctan2(Qe[1], Qe[2])  # Use arctan2 for better numerical stability

    # CREATE T3
    cos_psigsm, sin_psigsm = np.cos(-psigsm), np.sin(-psigsm)
    T3 = np.array(
        [[1, 0, 0], [0, cos_psigsm, sin_psigsm], [0, -sin_psigsm, cos_psigsm]]
    )

    return T1, T2, T3


def get_heliocentric_transformation_matrices(time):
    # Convert timestamp and compute Julian & Modified Julian Date
    ts = pd.Timestamp(time)
    jd = ts.to_julian_date()
    mjd = int(jd - 2400000.5)  # Modified Julian Date
    T0 = (mjd - 51544.5) / 36525.0
    UT = ts.hour + ts.minute / 60.0 + ts.second / 3600.0  # UT in hours

    # Precompute constants and use numpy operations for efficiency
    deg_to_rad = np.pi / 180
    LAMBDA = 280.460 + 36000.772 * T0 + 0.04107 * UT
    M = 357.528 + 35999.050 * T0 + 0.04107 * UT

    # Compute λ_sun in radians directly
    M_rad = M * deg_to_rad
    lt2 = (
        LAMBDA + (1.915 - 0.0048 * T0) * np.sin(M_rad) + 0.020 * np.sin(2 * M_rad)
    ) * deg_to_rad

    # Compute S1 transformation matrix using direct numpy operations
    lt2_pi = lt2 + np.pi
    cos_lt2, sin_lt2 = np.cos(lt2_pi), np.sin(lt2_pi)
    S1 = np.array([[cos_lt2, sin_lt2, 0], [-sin_lt2, cos_lt2, 0], [0, 0, 1]])
    # Equation 13 calculations
    iota = 7.25 * deg_to_rad
    omega = (73.6667 + 0.013958 * ((mjd + 3242) / 365.25)) * deg_to_rad  # in radians
    lambda_omega = lt2 - omega
    theta = np.arctan(np.cos(iota) * np.tan(lambda_omega))

    # Compute the quadrant of theta using vectorized numpy calculations
    lambda_omega_deg = np.mod(lambda_omega, 2 * np.pi) * (180 / np.pi)
    x, y = np.cos(np.radians(lambda_omega_deg)), np.sin(np.radians(lambda_omega_deg))
    x_theta, y_theta = np.cos(theta), np.sin(theta)

    if x >= 0 and y >= 0:
        if x_theta >= 0 and y_theta >= 0:
            theta = theta - np.pi
        elif x_theta >= 0 and y_theta <= 0:
            theta = theta - np.pi / 2
        elif x_theta <= 0 and y_theta >= 0:
            theta = theta + np.pi / 2

    elif x <= 0 and y <= 0:
        if x_theta <= 0 and y_theta <= 0:
            theta = theta - np.pi
        elif x_theta >= 0 and y_theta <= 0:
            theta = theta + np.pi / 2
        elif x_theta <= 0 and y_theta >= 0:
            theta = theta - np.pi / 2

    elif x >= 0 and y <= 0:
        if x_theta >= 0 and y_theta >= 0:
            theta = theta + np.pi / 2
        elif x_theta <= 0 and y_theta <= 0:
            theta = theta - np.pi / 2
        elif x_theta >= 0 and y_theta <= 0:
            theta = theta - np.pi

    elif x < 0 and y > 0:
        if x_theta >= 0 and y_theta >= 0:
            theta = theta - np.pi / 2
        elif x_theta <= 0 and y_theta <= 0:
            theta = theta + np.pi / 2
        elif x_theta <= 0 and y_theta >= 0:
            theta = theta - np.pi

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosiota = np.cos(iota)
    siniota = np.sin(iota)
    cosomega = np.cos(omega)
    sinomega = np.sin(omega)

    s2_theta = np.array([[costheta, sintheta, 0], [-sintheta, costheta, 0], [0, 0, 1]])
    s2_iota = np.array([[1, 0, 0], [0, cosiota, siniota], [0, -siniota, cosiota]])
    s2_omega = np.array([[cosomega, sinomega, 0], [-sinomega, cosomega, 0], [0, 0, 1]])
    S2 = s2_theta @ s2_iota @ s2_omega

    return S1, S2


"""
Geocentric position conversions
"""


def GSE_to_GSM(df):
    B_GSM = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_geocentric_transformation_matrices(df["time"].iloc[0])
        B_GSE_i = np.matrix([[df["x"].iloc[i]], [df["y"].iloc[i]], [df["z"].iloc[i]]])
        B_GSM_i = np.dot(T3, B_GSE_i)
        B_GSM_i_list = B_GSM_i.tolist()
        flat_B_GSM_i = list(itertools.chain(*B_GSM_i_list))
        r, lat, lon = cart2sphere(flat_B_GSM_i[0], flat_B_GSM_i[1], flat_B_GSM_i[2])
        position = flat_B_GSM_i[0], flat_B_GSM_i[1], flat_B_GSM_i[2], r, lat, lon
        B_GSM.append(position)
    df_transformed = pd.DataFrame(B_GSM, columns=["x", "y", "z", "r", "lat", "lon"])
    df_transformed["time"] = df["time"]
    return df_transformed


def GSM_to_GSE(df):
    B_GSE = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_geocentric_transformation_matrices(df["time"].iloc[0])
        T3_inv = np.linalg.inv(T3)
        B_GSM_i = np.matrix([[df["x"].iloc[i]], [df["y"].iloc[i]], [df["z"].iloc[i]]])
        B_GSE_i = np.dot(T3_inv, B_GSM_i)
        B_GSE_i_list = B_GSE_i.tolist()
        flat_B_GSE_i = list(itertools.chain(*B_GSE_i_list))
        r, lat, lon = cart2sphere(flat_B_GSE_i[0], flat_B_GSE_i[1], flat_B_GSE_i[2])
        position = flat_B_GSE_i[0], flat_B_GSE_i[1], flat_B_GSE_i[2], r, lat, lon
        B_GSE.append(position)
    df_transformed = pd.DataFrame(B_GSE, columns=["x", "y", "z", "r", "lat", "lon"])
    df_transformed["time"] = df["time"]
    return df_transformed


"""
Heliocentric position conversions
"""


def HEE_to_HAE(df):
    B_HAE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df["time"].iloc[0])
        S1_inv = np.linalg.inv(S1)
        B_HEE_i = np.matrix([[df["x"].iloc[i]], [df["y"].iloc[i]], [df["z"].iloc[i]]])
        B_HEA_i = np.dot(S1_inv, B_HEE_i)
        B_HAE_i_list = B_HEA_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        r, lat, lon = cart2sphere(flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2])
        position = flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2], r, lat, lon
        B_HAE.append(position)
    df_transformed = pd.DataFrame(B_HAE, columns=["x", "y", "z", "r", "lat", "lon"])
    df_transformed["time"] = df["time"]
    return df_transformed


def HAE_to_HEE(df):
    B_HEE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df["time"].iloc[0])
        B_HAE_i = np.matrix([[df["x"].iloc[i]], [df["y"].iloc[i]], [df["z"].iloc[i]]])
        B_HEE_i = np.dot(S1, B_HAE_i)
        B_HEE_i_list = B_HEE_i.tolist()
        flat_B_HEE_i = list(itertools.chain(*B_HEE_i_list))
        r, lat, lon = cart2sphere(flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2])
        position = flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2], r, lat, lon
        B_HEE.append(position)
    df_transformed = pd.DataFrame(B_HEE, columns=["x", "y", "z", "r", "lat", "lon"])
    df_transformed["time"] = df["time"]
    return df_transformed


def HAE_to_HEEQ(df):
    B_HEEQ = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df["time"].iloc[0])
        B_HAE_i = np.matrix([[df["x"].iloc[i]], [df["y"].iloc[i]], [df["z"].iloc[i]]])
        B_HEEQ_i = np.dot(S2, B_HAE_i)
        B_HEEQ_i_list = B_HEEQ_i.tolist()
        flat_B_HEEQ_i = list(itertools.chain(*B_HEEQ_i_list))
        r, lat, lon = cart2sphere(flat_B_HEEQ_i[0], flat_B_HEEQ_i[1], flat_B_HEEQ_i[2])
        position = flat_B_HEEQ_i[0], flat_B_HEEQ_i[1], flat_B_HEEQ_i[2], r, lat, lon
        B_HEEQ.append(position)
    df_transformed = pd.DataFrame(B_HEEQ, columns=["x", "y", "z", "r", "lat", "lon"])
    df_transformed["time"] = df["time"]
    return df_transformed


def HEEQ_to_HAE(df):
    B_HAE = []
    for i in range(df.shape[0]):
        S1, S2 = get_heliocentric_transformation_matrices(df["time"].iloc[0])
        S2_inv = np.linalg.inv(S2)
        B_HEEQ_i = np.matrix([[df["x"].iloc[i]], [df["y"].iloc[i]], [df["z"].iloc[i]]])
        B_HEA_i = np.dot(S2_inv, B_HEEQ_i)
        B_HAE_i_list = B_HEA_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        r, lat, lon = cart2sphere(flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2])
        position = flat_B_HAE_i[0], flat_B_HAE_i[1], flat_B_HAE_i[2], r, lat, lon
        B_HAE.append(position)
    df_transformed = pd.DataFrame(B_HAE, columns=["x", "y", "z", "r", "lat", "lon"])
    df_transformed["time"] = df["time"]
    return df_transformed


def HEE_to_HEEQ(df):
    df_hae = HEE_to_HAE(df)
    df_transformed = HAE_to_HEEQ(df_hae)
    return df_transformed


def HEEQ_to_HEE(df):
    df_hae = HEEQ_to_HAE(df)
    df_transformed = HAE_to_HEE(df_hae)
    return df_transformed


"""
Geocentric to heliocentric position conversions
#requires extra step in the conversion of GSE to HEE i.e. adding position vector of Sun
"""


def get_rsun_position_vector(time):
    # format dates correctly, calculate MJD, T0, UT
    ts = pd.Timestamp(time)
    jd = ts.to_julian_date()
    mjd = float(int(jd - 2400000.5))  # use modified julian date
    T0 = (mjd - 51544.5) / 36525.0
    UT = ts.hour + ts.minute / 60.0 + ts.second / 3600.0  # time in UT in hours
    LAMBDA = 280.460 + 36000.772 * T0 + 0.04107 * UT
    M = 357.528 + 35999.050 * T0 + 0.04107 * UT
    lt2 = (
        (
            LAMBDA
            + (1.915 - 0.0048 * T0) * np.sin(M * np.pi / 180)
            + 0.020 * np.sin(2 * M * np.pi / 180)
        )
        * np.pi
        / 180
    )  # lamda sun
    # section 6.1
    r_0 = 1.495985e8  # units km
    e = 0.016709 - 0.0000418 * T0
    omega_bar = 282.94 + 1.72 * T0
    v = lt2 - omega_bar
    # final r_sun equation
    r_sun = (r_0 * (1 - e**2)) / (1 + e * np.cos(v))
    R_sun = np.matrix([[r_sun], [0], [0]])
    return R_sun


def GSE_to_HEE(df):
    B_HEE = []
    z_rot_180 = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    for i in range(df.shape[0]):
        R_sun = get_rsun_position_vector(df["time"].iloc[0])
        try:
            B_GSE_i = np.matrix(
                [[df["x"].iloc[i]], [df["y"].iloc[i]], [df["z"].iloc[i]]]
            )
        except:
            B_GSE_i = np.matrix(
                [
                    [df["GSE_X(km)"].iloc[i]],
                    [df["GSE_y(km)"].iloc[i]],
                    [df["GSE_z(km)"].iloc[i]],
                ]
            )
        B_HEE_i = R_sun + np.dot(z_rot_180, B_GSE_i)
        B_HEE_i_list = B_HEE_i.tolist()
        flat_B_HEE_i = list(itertools.chain(*B_HEE_i_list))
        r, lat, lon = cart2sphere(flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2])
        position = flat_B_HEE_i[0], flat_B_HEE_i[1], flat_B_HEE_i[2], r, lat, lon
        B_HEE.append(position)
    df_transformed = pd.DataFrame(B_HEE, columns=["x", "y", "z", "r", "lat", "lon"])
    df_transformed["time"] = df["time"]
    return df_transformed
