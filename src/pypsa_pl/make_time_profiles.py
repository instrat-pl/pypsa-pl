import pandas as pd
import numpy as np
import logging

from pypsa_pl.config import data_dir
from pypsa_pl.mathematical_operations import modify_vres_availability_profile


def set_snapshot_index(df, params):
    df = df.rename(columns={"hour": "timestep"})
    df["timestep"] = df["timestep"].str.replace(
        str(params["weather_year"]), str(params["year"])
    )
    df = df.set_index("timestep")
    return df


def make_electricity_final_use_load_profile(df, snapshots, params):
    df_t = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"demand_profile;carrier=electricity final use;year={params['weather_year']}.csv",
        )
    )
    # Create dataframe with profiles for each final use component based on area match
    df_t = set_snapshot_index(df_t, params).loc[snapshots]
    df_t = df_t.transpose().reset_index(names="area")
    df_t = (
        df[["name", "area", "p_set_annual"]]
        .merge(df_t, on="area", how="inner")
        .drop(columns="area")
    )
    # Multiply the load profile by p_set_annual
    df_t[snapshots] *= df_t["p_set_annual"].values[:, np.newaxis]
    df_t = df_t.drop(columns="p_set_annual")
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_public_heat_final_use_load_profile(df, snapshots, params):
    df_t = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"demand_profile;carrier=space heating final use;year={params['weather_year']}.csv",
        )
    )
    df_t = set_snapshot_index(df_t, params).loc[snapshots]
    df_t = params["share_space_heating"] * df_t + (1 - params["share_space_heating"])
    df_t = df_t.transpose().reset_index(names="area")
    df_t = (
        df[["name", "area", "p_set_annual", "p_nom"]]
        .merge(df_t, on="area", how="inner")
        .drop(columns="area")
    )
    # Multiply the load profile by p_set_annual
    df_t[snapshots] *= df_t["p_set_annual"].values[:, np.newaxis]
    # TODO: find ways of dealing with p_nom > p_set_annual preserving utilisation factor
    df_t[snapshots] = np.minimum(df_t[snapshots], df_t["p_nom"].values[:, np.newaxis])
    df_t = df_t.drop(columns=["p_set_annual", "p_nom"])
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_final_use_load_pu_profile_func(carrier):
    def make_final_use_load_pu_profile(df, snapshots, params):
        df_t = pd.read_csv(
            data_dir(
                "input",
                f"timeseries;variant={params['timeseries']}",
                f"demand_profile;carrier={carrier};year={params['weather_year']}.csv",
            )
        )
        df_t = set_snapshot_index(df_t, params).loc[snapshots]
        df_t = df_t.transpose().reset_index(names="area")
        df_t = (
            df[["name", "area", "p_set_pu_annual"]]
            .merge(df_t, on="area", how="inner")
            .drop(columns="area")
        )
        # Multiply the load profile by p_set_pu_annual over mean load p.u.
        df_t[snapshots] *= df_t["p_set_pu_annual"].values[:, np.newaxis]
        # TODO: find ways of dealing with p_set_pu > 1 preserving utilisation factor
        df_t[snapshots] = np.minimum(df_t[snapshots], 1)
        df_t = df_t.drop(columns=["p_set_pu_annual"])
        df_t = df_t.set_index("name").transpose()
        return df_t

    return make_final_use_load_pu_profile


make_space_heating_final_use_load_pu_profile = make_final_use_load_pu_profile_func(
    "space heating final use"
)
make_light_vehicle_mobility_final_use_load_pu_profile = (
    make_final_use_load_pu_profile_func("light vehicle mobility final use")
)


def make_vres_availability_profile(df, snapshots, params):
    dfs_t = {
        carrier: pd.read_csv(
            data_dir(
                "input",
                f"timeseries;variant={params['timeseries']}",
                f"availability_profile;carrier={carrier};year={params['weather_year']}.csv",
            )
        )
        for carrier in df["carrier"].unique()
    }
    df_t = pd.concat(
        [
            set_snapshot_index(df_t, params)
            .loc[snapshots]
            .transpose()
            .reset_index(names="area")
            .assign(**{"carrier": carrier})
            for carrier, df_t in dfs_t.items()
        ]
    )
    df_t = (
        df[["name", "area", "carrier", "qualifier", "p_max_pu_annual"]]
        .merge(df_t, on=["area", "carrier"], how="inner")
        .drop(columns=["carrier"])
    )

    # Modify availability profiles s.t. they match the assumed annual availability factors
    # Only for PL
    is_domestic = df_t["area"].str.startswith("PL")
    df_t.loc[is_domestic, snapshots] = modify_vres_availability_profile(
        df_t.loc[is_domestic, snapshots].values.transpose(),
        annual_availability_factor=df_t.loc[is_domestic, "p_max_pu_annual"].values,
    ).transpose()

    # For prosumer vRES capacities, reduce the availability by self consumption rate
    is_prosumer = df_t["qualifier"] == "prosumer"
    df_t.loc[is_prosumer, snapshots] *= 1 - params["prosumer_self_consumption"]

    df_t = df_t.drop(columns=["area", "qualifier", "p_max_pu_annual"])
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_constant_load_profile(df, snapshots, params):
    df = df.set_index("name")
    df_t = pd.DataFrame(1, index=snapshots, columns=df.index)
    df_t *= df["p_set"]
    return df_t


def make_constant_load_pu_profile(df, snapshots, params):
    df = df.set_index("name")
    df_t = pd.DataFrame(1, index=snapshots, columns=df.index)
    df_t *= df["p_set_pu"].fillna(df["p_set_pu_annual"])
    return df_t


def make_heat_pump_cop_profile(df, snapshots, params):
    dfs_t = {
        technology: pd.read_csv(
            data_dir(
                "input",
                f"timeseries;variant={params['timeseries']}",
                f"cop_profile;technology={technology};year={params['weather_year']}.csv",
            )
        )
        for technology in df["technology"].unique()
    }
    df_t = pd.concat(
        [
            set_snapshot_index(df_t, params)
            .loc[snapshots]
            .transpose()
            .reset_index(names="area")
            .assign(**{"technology": technology})
            for technology, df_t in dfs_t.items()
        ]
    )
    df_t = (
        df[["name", "area", "technology"]]
        .merge(df_t, on=["area", "technology"], how="inner")
        .drop(columns=["area", "technology"])
    )
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_heat_pump_max_output_pu_profile(df, snapshots, params):
    dfs_t = {
        technology: pd.read_csv(
            data_dir(
                "input",
                f"timeseries;variant={params['timeseries']}",
                f"cop_profile;technology={technology};year={params['weather_year']}.csv",
            )
        )
        for technology in df["technology"].unique()
    }
    df_t = pd.concat(
        [
            set_snapshot_index(df_t, params)
            .loc[snapshots]
            .transpose()
            .reset_index(names="area")
            .assign(**{"technology": technology})
            for technology, df_t in dfs_t.items()
        ]
    )
    df_t = (
        df[["name", "area", "technology", "efficiency"]]
        .merge(df_t, on=["area", "technology"], how="inner")
        .drop(columns=["area", "technology"])
    )
    # Divide the actual hourly COP by the nominal COP to obtain p_max_pu
    df_t[snapshots] /= df_t["efficiency"].values[:, np.newaxis]
    # p_max_pu is limited to 1
    df_t[snapshots] = np.minimum(df_t[snapshots], 1)
    df_t = df_t.drop(columns=["efficiency"])
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_bev_charger_max_output_pu_profile(df, snapshots, params):
    df_t = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"demand_profile;carrier=light vehicle mobility final use;year={params['weather_year']}.csv",
        )
    )
    df_t = set_snapshot_index(df_t, params).loc[snapshots]
    df_t = df_t.transpose().reset_index(names="area")
    df_t = df[["name", "area"]].merge(df_t, on="area", how="inner").drop(columns="area")
    # Based on PyPSA-Eur approach
    # https://github.com/PyPSA/pypsa-eur/blob/v0.11.0/scripts/build_transport_demand.py#L120
    bev_availability_max = params["bev_availability_max"]
    bev_availability_mean = params["bev_availability_mean"]
    profile_min = df_t[snapshots].min(axis=1).values[:, np.newaxis]
    profile_mean = df_t[snapshots].mean(axis=1).values[:, np.newaxis]

    df_t[snapshots] = bev_availability_max - (
        bev_availability_max - bev_availability_mean
    ) * (df_t[snapshots] - profile_min) / (profile_mean - profile_min)
    df_t[snapshots] = np.maximum(df_t[snapshots], 0)

    df_t = df_t.set_index("name").transpose()
    return df_t


def make_bev_charger_min_output_pu_profile(df, snapshots, params):
    # p_min_pu is larger than 0 if the assumed share of inflexible BEV charging is larger than 0
    bev_inflexible_share = 1 - params.get("bev_flexible_share", 1)

    df_t = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"bev_inflexible_charging_profile;year={params['weather_year']}.csv",
        )
    )
    df_t = set_snapshot_index(df_t, params).loc[snapshots]
    df_t = df_t.transpose().reset_index(names="area")
    df_t = df[["name", "area"]].merge(df_t, on="area", how="inner").drop(columns="area")

    bev_utilisation = params["light_vehicle_mobility_utilisation"]
    # 10 kW of avg. max wheel power and 9.9 (0.9 * 11) kWh of charger capacity per BEV
    bev_output_to_charger_ratio = params.get("bev_output_to_charger_ratio", 10 / 9.9)
    # Battery-to-wheel efficiency
    bev_efficiency = params.get("bev_efficiency", 0.85)

    df_t[snapshots] *= (
        bev_inflexible_share
        * bev_utilisation
        / bev_efficiency
        * bev_output_to_charger_ratio
    )

    df_t = df_t.set_index("name").transpose()
    return df_t


def calculate_bev_battery_max_soc_profile(snapshots, params):
    # (0) Define parameters
    bev_utilisation = params["light_vehicle_mobility_utilisation"]
    # TODO: if possible, get them from the technology cost data
    # 10 kW of avg. max wheel power and 44 kWh of battery capacity per BEV
    bev_output_to_battery_ratio = params.get("bev_output_to_battery_ratio", 10 / 44)
    # Battery-to-wheel efficiency
    bev_efficiency = params.get("bev_efficiency", 0.85)

    # Assume certain max state of charge level of the BEV fleet - if high enough, it will not influence the result
    soc_max = params.get("bev_battery_max_soc", 0.8)

    # (1) Calculate e_max_pu series based on inflexible charging and consumption profiles
    df_t_discharge = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"demand_profile;carrier=light vehicle mobility final use;year={params['weather_year']}.csv",
        )
    )
    df_t_charge = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"bev_inflexible_charging_profile;year={params['weather_year']}.csv",
        )
    )
    df_t_discharge = set_snapshot_index(df_t_discharge, params).loc[snapshots]
    df_t_charge = set_snapshot_index(df_t_charge, params).loc[snapshots]
    df_t = (
        (df_t_charge - df_t_discharge).cumsum()
        * bev_output_to_battery_ratio
        * bev_utilisation
        / bev_efficiency
    )
    df_t += soc_max - df_t.max()
    return df_t


def make_bev_battery_max_soc_profile(df, snapshots, params):
    df_t = calculate_bev_battery_max_soc_profile(snapshots, params)

    df_t = df_t.transpose().reset_index(names="area")
    df_t = df[["name", "area"]].merge(df_t, on="area", how="inner").drop(columns="area")

    df_t = df_t.set_index("name").transpose()
    return df_t


def make_bev_battery_min_soc_profile(df, snapshots, params):
    # e_max_t - e_min_t timeseries determine the flexibility of the BEV storage
    # the flexibility is limited due to different charging and discharging patterns of individual BEVs
    # approach inspired by Muessel et al. 2023 (https://doi.org/10.1016/j.isci.2023.107816)
    # (0) Define parameters
    bev_utilisation = params["light_vehicle_mobility_utilisation"]
    # TODO: if possible, get them from the technology cost data
    # 10 kW of avg. max wheel power and 44 kWh of battery capacity per BEV
    bev_output_to_battery_ratio = params.get("bev_output_to_battery_ratio", 10 / 44)
    # Battery-to-wheel efficiency
    bev_efficiency = params.get("bev_efficiency", 0.85)
    # Assume average flexibility is of the order of the daily electricity consumption of BEVs
    snapshots_per_day = 24
    e_flex_pu_mean = (
        bev_output_to_battery_ratio
        * bev_utilisation
        / bev_efficiency
        * snapshots_per_day
        * params.get("bev_flexibility_factor", 1)
    )
    logging.info(
        f"Mean BEV storage flexibility: {e_flex_pu_mean * 44:.2f} kWh/BEV (flexible vehicles only)"
    )
    # Assume the ratio of mean to max flexibility is specified by the user
    # Default ratio stems from Muessel et al. 2023 (https://doi.org/10.1016/j.isci.2023.107816)
    e_flex_pu_max = (
        params.get("bev_flexibility_max_to_mean_ratio", 1.33) * e_flex_pu_mean
    )

    # (1) Calculate e_flex_pu_t assuming it is related to BEV battery consumption
    # i.e. the higher the consumption, the lower the flexibility
    df_flex_t = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"demand_profile;carrier=light vehicle mobility final use;year={params['weather_year']}.csv",
        )
    )
    df_flex_t = set_snapshot_index(df_flex_t, params).loc[snapshots]
    df_flex_t = df_flex_t.transpose().reset_index(names="area")

    profile_min = df_flex_t[snapshots].min(axis=1).values[:, np.newaxis]
    profile_mean = df_flex_t[snapshots].mean(axis=1).values[:, np.newaxis]
    df_flex_t[snapshots] = e_flex_pu_max - (e_flex_pu_max - e_flex_pu_mean) * (
        df_flex_t[snapshots] - profile_min
    ) / (profile_mean - profile_min)

    df_flex_t[snapshots] *= params.get("bev_flexible_share", 1)

    df_flex_t = df_flex_t.set_index("area").transpose()

    # (2) Subtract e_flex_pu_t from e_max_pu_t to obtain e_min_pu_t
    df_max_t = calculate_bev_battery_max_soc_profile(snapshots, params)
    df_t = df_max_t - df_flex_t

    df_t = df_t.transpose().reset_index(names="area")
    df_t = df[["name", "area"]].merge(df_t, on="area", how="inner").drop(columns="area")

    df_t = df_t.set_index("name").transpose()
    return df_t


make_profile_funcs = {
    "electricity final use load profile": make_electricity_final_use_load_profile,
    "public heat final use load profile": make_public_heat_final_use_load_profile,
    "space heating final use load pu profile": make_space_heating_final_use_load_pu_profile,
    "light vehicle mobility final use load pu profile": make_light_vehicle_mobility_final_use_load_pu_profile,
    "vres availability profile": make_vres_availability_profile,
    "constant load profile": make_constant_load_profile,
    "constant load pu profile": make_constant_load_pu_profile,
    "heat pump COP profile": make_heat_pump_cop_profile,
    "heat pump max output pu profile": make_heat_pump_max_output_pu_profile,
    "BEV charger max output pu profile": make_bev_charger_max_output_pu_profile,
    "BEV charger min output pu profile": make_bev_charger_min_output_pu_profile,
    "BEV battery max SOC profile": make_bev_battery_max_soc_profile,
    "BEV battery min SOC profile": make_bev_battery_min_soc_profile,
}
