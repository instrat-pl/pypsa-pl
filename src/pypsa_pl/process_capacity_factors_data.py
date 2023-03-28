import pandas as pd
import numpy as np

from pypsa_pl.config import data_dir
from pypsa_pl.io import read_excel
from pypsa_pl.helper_functions import repeat_over_periods


def modify_wind_utilization_profile(profile, annual_correction_factor):
    """
    cf --> cf * f(cf)
    f(cf) is the correction factor; it is zero at cf = 1
    f(cf) = a + (1 - a) * (2 * cf - 1)
    a is the correction factor at cf = 0.5
    a = (ACF * < cf > - < cf * (2 * cf - 1) >) / (< cf > - < cf * (2 * cf - 1) >)
    """
    x = np.mean(profile)
    y = np.mean(profile * (2 * profile - 1))

    a = (annual_correction_factor * x - y) / (x - y)
    profile = profile * (a + (1 - a) * (2 * profile - 1))

    assert np.abs(np.mean(profile) - annual_correction_factor * x) < 1e-5
    return profile


def process_utilization_profiles(
    source_renewable,
    source_chp,
    network,
    weather_year,
    temporal_resolution,
    correction_factor_wind_old=1,
    correction_factor_wind_new=1,
    domestic=True,
):
    dfs = {}
    for cat in ["CHP", "PV", "Wind offshore", "Wind onshore", "Wind onshore old"]:
        if not domestic and cat == "Wind onshore old":
            continue

        source = source_renewable if cat != "CHP" else source_chp
        prefix = cat.lower().replace(" ", "_")

        # Use modified wind onshore for wind onshore old for non-PyPSA-PL v1 sources
        create_wind_onshore_old_cf = (
            cat == "Wind onshore old" and source != "pypsa_pl_v1"
        )
        if create_wind_onshore_old_cf:
            prefix = "wind_onshore"

        file = data_dir("input", f"{prefix}_utilization_profile;source={source}.xlsx")

        # Ignore weather year for PyPSA-PL v1 source
        sheet_name = str(weather_year) if source != "pypsa_pl_v1" else 0

        column_filter = (
            lambda col: (col == "hour")
            or (col == "Fuel")
            or (col.startswith("PL") if domestic else not col.startswith("PL"))
        )

        df = read_excel(file, sheet_name=sheet_name, usecols=column_filter)

        if cat.startswith("Wind onshore"):
            df = df.set_index("hour")
            for col in df.columns:
                df[col] = modify_wind_utilization_profile(
                    df[col],
                    correction_factor_wind_new
                    if cat == "Wind onshore"
                    else correction_factor_wind_old,
                )
            df = df.reset_index()

        if "Fuel" in df.columns:
            df = df.melt(
                id_vars=["hour", "Fuel"],
                var_name="Area",
                value_name="cf",
            )
            df = df.pivot(
                index="hour",
                columns=["Area", "Fuel"],
                values="cf",
            ).reset_index()
        # Aggregate the capacity factors timeseries to the temporal resolution of the snapshots
        df["hour"] = pd.to_datetime(df["hour"])
        df = (
            df.groupby(pd.Grouper(key="hour", freq=temporal_resolution))
            .sum()
            .reset_index()
        )
        df = df.rename(columns={"hour": "timestep"})
        df = df.set_index("timestep").round(3).reset_index()
        df = repeat_over_periods(df, network)
        dfs[cat] = df
    return dfs
