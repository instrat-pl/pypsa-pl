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
