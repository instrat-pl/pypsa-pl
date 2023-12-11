import numpy as np
import pandas as pd


def select_subset(df, var, vals):
    all_vals = set(df[var])
    vals = [val for val in vals if val in all_vals]
    df = pd.merge(df, pd.Series(vals, name=var), on=var, how="right")
    return df


def repeat_over_periods(df, network):
    df_snapshots = pd.DataFrame(index=network.snapshots).reset_index()
    grouper = {
        "month": lambda df: df.timestep.dt.month,
        "day": lambda df: df.timestep.dt.day,
        "hour": lambda df: df.timestep.dt.hour,
    }
    df = pd.merge(
        df_snapshots.assign(**grouper),
        df.assign(**grouper).drop(columns="timestep"),
        on=[col for col in grouper.keys()],
        how="left",
    ).drop(columns=[col for col in grouper.keys()])
    return df


def repeat_over_timesteps(df, network):
    df_snapshots = pd.DataFrame(index=network.snapshots).reset_index()
    df = pd.merge(
        df_snapshots,
        df,
        left_on="period",
        right_on="year",
        how="left",
    ).drop(columns="year")
    return df


def calculate_annuity(lifetime, discount_rate):
    if discount_rate > 0:
        return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)
    else:
        return 1 / lifetime


def filter_lifetimes(df, years):
    year_min, year_max = min(years), max(years)
    df = df[
        (df["build_year"] <= year_min) & (year_max < df["build_year"] + df["lifetime"])
    ]
    return df


def update_lifetime(df, default_build_year, decommission_year_inclusive):
    # Determine build_year and lifetime parameters
    if "build_year" not in df.columns:
        df["build_year"] = default_build_year
    else:
        df["build_year"] = df["build_year"].fillna(default_build_year)
    if "retire_year" not in df.columns:
        df["retire_year"] = np.inf
    else:
        df["retire_year"] = df["retire_year"].fillna(np.inf)
    df["lifetime"] = df["retire_year"] - df["build_year"]
    if decommission_year_inclusive:
        df["lifetime"] += 1
    return df


# Statistics


def get_attr(attr):
    def getter(n, c):
        df = n.df(c)
        if attr in df:
            values = df[attr].fillna("")
        else:
            values = pd.Series("", index=df.index)
        return values.rename(attr)

    return getter


def get_attrs(attrs):
    def getter(n, c):
        return [get_attr(attr)(n, c) for attr in attrs]

    return getter


def get_bus_attr(bus, attr):
    def getter(n, c):
        df = n.df(c)
        if bus in df:
            values = df[bus].map(n.buses[attr]).fillna("")
        else:
            values = pd.Series("", index=df.index)
        return values.rename(f"{bus}_{attr}")

    return getter


def custom_groupby(n, c):
    return (
        [get_attr(attr)(n, c) for attr in ["area", "sector", "bus", "bus0", "bus1"]]
        + [get_bus_attr(bus, "carrier")(n, c) for bus in ["bus", "bus0", "bus1"]]
        + [
            get_attr(attr)(n, c)
            for attr in [
                "build_year",
                "lifetime",
                "p0_sign",
                "category",
                "carrier",
                "technology",
            ]
        ]
    )


def calculate_statistics(network):
    df = (
        network.statistics(
            groupby=custom_groupby,
            aggregate_time="mean",
        )
        .reset_index()
        .rename(columns={"level_0": "component"})
    )
    return df
