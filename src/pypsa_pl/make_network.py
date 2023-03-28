import pandas as pd
import pypsa

from pypsa_pl.ignore_warnings import ignore_future_warnings


def make_snapshots(years, freq="1H", leap_days="remove"):
    snapshots = pd.DatetimeIndex([], name="snapshot")
    for year in years:
        s = pd.date_range(
            start=f"{year}-01-01 00:00",
            end=f"{year+1}-01-01 00:00",
            freq=freq,
            inclusive="left",
        )
        if leap_days == "remove":
            s = s[~((s.month == 2) & (s.day == 29))]
        snapshots = snapshots.append(s)
    snapshots = pd.MultiIndex.from_arrays(
        [snapshots.year, snapshots], names=["year", "snapshot"]
    )
    return snapshots


@ignore_future_warnings
def make_network(
    temporal_resolution="1H", years=[2020, 2030, 2040, 2050], discount_rate=0.03
):
    network = pypsa.Network()
    network.set_snapshots(make_snapshots(years, freq=temporal_resolution))
    network.snapshot_weightings.loc[:, :] = int(
        temporal_resolution[:-1]
    )  # TODO: check if this suffices to change the temporal resolution

    network.set_investment_periods(years)
    if len(years) > 1:
        network.investment_period_weightings["years"] = (
            pd.Series(years, index=years)
            .diff()
            .shift(-1)
            .fillna(method="pad")
            .astype(int)
        )
        year_count = 0
        for period, n_years in network.investment_period_weightings["years"].items():
            network.investment_period_weightings.loc[period, "objective"] = sum(
                1 / (1 + discount_rate) ** y
                for y in range(year_count, year_count + n_years)
            )
            year_count += n_years

    return network
