import pandas as pd
import numpy as np

from pypsa_pl.config import data_dir
from pypsa_pl.io import read_excel
from pypsa_pl.process_technology_data import load_technology_data, get_technology_year
from pypsa_pl.helper_functions import calculate_annuity, update_lifetime


def assign_reserve_assumptions(
    df,
    primary_reserve_categories=None,
    tertiary_reserve_categories=None,
    primary_reserve_minutes=1,
    tertiary_reserve_minutes=30,
    cold_reserve_categories=None,
):
    is_domestic = df["area"].str.startswith("PL")

    df["is_primary_reserve"] = False
    if primary_reserve_categories:
        df.loc[
            is_domestic & df["category"].isin(primary_reserve_categories),
            "is_primary_reserve",
        ] = True
        for direction in ["up", "down"]:
            df.loc[
                df["is_primary_reserve"], f"primary_reserve_ramp_limit_{direction}"
            ] = (
                primary_reserve_minutes
                * df.loc[
                    df["is_primary_reserve"],
                    f"Ramp {direction} rate limit [% of max output / min]",
                ]
            )
            df.loc[
                df["is_primary_reserve"]
                & ~(df[f"primary_reserve_ramp_limit_{direction}"] <= 1),
                f"primary_reserve_ramp_limit_{direction}",
            ] = 1.0
    df["is_tertiary_reserve"] = False
    if tertiary_reserve_categories:
        df.loc[
            is_domestic & df["category"].isin(tertiary_reserve_categories),
            "is_tertiary_reserve",
        ] = True
        df.loc[df["is_tertiary_reserve"], f"tertiary_reserve_ramp_limit_up"] = (
            tertiary_reserve_minutes
            * df.loc[
                df["is_tertiary_reserve"],
                f"Ramp up rate limit [% of max output / min]",
            ]
        )
        df.loc[
            df["is_tertiary_reserve"] & ~(df[f"tertiary_reserve_ramp_limit_up"] <= 1),
            f"tertiary_reserve_ramp_limit_up",
        ] = 1.0

    df["is_cold_reserve"] = False
    if cold_reserve_categories:
        df.loc[
            is_domestic & df["category"].isin(cold_reserve_categories),
            "is_cold_reserve",
        ] = True

    is_warm_reserve = df["is_primary_reserve"] | df["is_tertiary_reserve"]
    df.loc[is_warm_reserve, "p_min_pu_stable"] = (
        df.loc[is_warm_reserve, "Minimum generation [% of max output]"]
        * df.loc[is_warm_reserve, "p_max_pu"].fillna(1)
    ).fillna(0)
    return df


def process_utility_units_data(
    source_thermal,
    source_renewable,
    source_storage,
    source_technology,
    default_build_year=2020,
    decommission_year_inclusive=True,
    unit_commitment_categories=None,
    linearized_unit_commitment=True,
    thermal_constraints_factor=1,
    hours_per_timestep=1,
):
    """
    Process data on individual utility units (thermal, renewable, storage).
    """

    dfs = []

    for prefix, source in [
        ("thermal", source_thermal),
        ("renewable", source_renewable),
        ("storage", source_storage),
    ]:
        if source is None:
            continue
        df = read_excel(
            data_dir("input", f"{prefix}_units;source={source}.xlsx"),
            sheet_var="group",
        ).drop(columns="group")

        include_col = "is_active"
        if include_col in df.columns:
            df = df[df[include_col]].drop(columns=include_col)

        # Assume build year for all units not earlier than 2000
        df.loc[df["build_year"] < default_build_year, "build_year"] = default_build_year

        df = update_lifetime(
            df,
            default_build_year=default_build_year,
            decommission_year_inclusive=decommission_year_inclusive,
        )

        # Infer technology data

        # Match the build year to technology year for 5 year investment periods
        # e.g. 2020 -> 2015, 2023 -> 2020, 2025 -> 2020, 2026 -> 2025
        df["technology_year"] = get_technology_year(df["build_year"]).astype(int)

        # Fill missing parameters by default technology data
        technology_years = df["technology_year"].drop_duplicates().tolist()
        df_tech = load_technology_data(source=source_technology, years=technology_years)
        df = pd.merge(
            df,
            df_tech,
            on=["technology_year", "technology"],
            how="left",
        )

        df["p_nom"] = df["p_nom"].round(3)
        df["p_max_pu"] = df["Net to gross power ratio"].round(2).fillna(1)
        # Custom parameter p_mean_max_pu = maximum annual capacity utilization factor
        df["Maximum utilization due to outages"] = (
            1
            - df["Planned outage as year fraction"].fillna(0)
            - df["Forced outage as year fraction"].fillna(0)
        )
        df["p_max_pu_annual"] = (
            df["p_max_pu"] * df["Maximum utilization due to outages"]
        ).round(3)

        # For wind and PV weather is a stronger bound so ignore
        df.loc[
            df["carrier"].str.startswith("Wind") | df["carrier"].str.startswith("PV"),
            "p_max_pu_annual",
        ] = 1

        # thermal unit assumptions
        if prefix == "thermal":
            df["efficiency"] = df["efficiency"].fillna(df["Net electrical efficiency"])

        # Renewable unit assumptions
        if prefix == "renewable":
            df["efficiency"] = 1.0

        # Storage unit assumptions
        if prefix == "storage":
            df["max_hours"] = df["charge_max"] / df["p_nom"]
            df["max_hours"] = (
                df["max_hours"].fillna(1 / df["Discharging power [MW/MWh]"]).round(3)
            )
            df["efficiency_round"] = df["efficiency_round"].fillna(
                df["Round trip efficiency"]
            )
            sqrt_efficiency = np.round(np.sqrt(df["efficiency_round"]), 3)
            df["efficiency_store"] = sqrt_efficiency
            df["efficiency_dispatch"] = sqrt_efficiency
            df["standing_loss"] = (df["Standing loss per day"] / 24).round(5).fillna(0)
            if "p_nom_store" in df.columns:
                df["p_min_pu"] = (-1) * np.minimum(
                    (df["p_nom_store"] / df["p_nom"]).round(3), 1
                )

        # For fixed units don't calculate the capital cost
        df["capital_cost"] = 0

        # Capacity of individual units is not to be extended
        df["p_nom_extendable"] = False

        if unit_commitment_categories:
            df["committable"] = False
            is_committable = df["category"].isin(unit_commitment_categories) & df[
                "area"
            ].str.startswith("PL")
            df.loc[is_committable, "committable"] = True
            # Split input startup cost into PyPSA's shutdown_cost and startup_cost
            df.loc[is_committable, "start_up_cost"] = (
                df.loc[is_committable, "Startup cost [PLN/MW]"]
                * df.loc[is_committable, "p_nom"]
                * thermal_constraints_factor
                / 2
            ).fillna(0)
            df.loc[is_committable, "shut_down_cost"] = df.loc[
                is_committable, "start_up_cost"
            ]
            df.loc[is_committable, "min_up_time"] = (
                (
                    df.loc[is_committable, "Minimum time on [hours]"]
                    / hours_per_timestep
                    * thermal_constraints_factor
                )
                .round()
                .astype(int)
                .fillna(0)
            )
            df.loc[is_committable, "min_down_time"] = (
                (
                    df.loc[is_committable, "Minimum time off [hours]"]
                    / hours_per_timestep
                    * thermal_constraints_factor
                )
                .round()
                .astype(int)
                .fillna(0)
            )
            df.loc[is_committable, "ramp_limit_start_up"] = (
                df.loc[
                    is_committable,
                    "Ramp up rate limit at startup [% of max output / hour]",
                ]
                * hours_per_timestep
                / thermal_constraints_factor
            ).fillna(1)
            df.loc[is_committable, "ramp_limit_shut_down"] = (
                df.loc[
                    is_committable,
                    "Ramp down rate limit at shutdown [% of max output / hour]",
                ]
                * hours_per_timestep
                / thermal_constraints_factor
            ).fillna(1)
            df.loc[is_committable, "ramp_limit_up"] = (
                df.loc[is_committable, "Ramp up rate limit [% of max output / min]"]
                * 60
                * hours_per_timestep
                / thermal_constraints_factor
            ).fillna(np.nan)
            df.loc[is_committable, "ramp_limit_down"] = (
                df.loc[is_committable, "Ramp down rate limit [% of max output / min]"]
                * 60
                * hours_per_timestep
                / thermal_constraints_factor
            ).fillna(np.nan)
            # If ramp limits are higher than 1, set them to 1
            for attr in [
                "ramp_limit_up",
                "ramp_limit_down",
                "ramp_limit_start_up",
                "ramp_limit_shut_down",
            ]:
                df.loc[df[attr] > 1, attr] = 1

            df.loc[is_committable, "p_min_pu"] = (
                df.loc[is_committable, "Minimum generation [% of max output]"]
                * df.loc[is_committable, "p_max_pu"].fillna(1)
            ).fillna(0)
            # For MILP problem, ramp limits for start up and shut down cannot be smaller than p_min_pu
            if not linearized_unit_commitment:
                for attr in ["ramp_limit_start_up", "ramp_limit_shut_down"]:
                    condition = is_committable & (df[attr] < df["p_min_pu"])
                    df.loc[condition, attr] = df.loc[condition, "p_min_pu"]

        df = df.dropna(axis=1, how="all")
        dfs.append(df)

    df = pd.concat(dfs)
    return df


def process_aggregate_capacity_data(
    df,
    source_technology,
    source_hydro_utilization,
    discount_rate=0,
    industrial_utilization=0.5,
    enforce_bio=0,
    extendable_technologies=None,
    default_build_year=2020,
    decommission_year_inclusive=True,
    active_investment_years=None,
    extend_from_zero=False,
):
    """
    Process data on aggregate generation and storage capacities by year and by area (voivodeship or country).
    """

    # Round the values
    df["p_nom"] = df["p_nom"].round(3)

    # Load default technological assumptions
    df["technology_year"] = get_technology_year(df["build_year"]).astype(int)
    technology_years = df["technology_year"].drop_duplicates().tolist()
    df_tech = load_technology_data(source=source_technology, years=technology_years)
    df = pd.merge(df, df_tech, on=["technology_year", "technology"], how="left")

    # Incorporate hydro utilization inputs as p_max_pu (PL) or p_max_pu_annual (neighbors)
    df_hydro = read_excel(
        data_dir("input", f"hydro_utilization;source={source_hydro_utilization}.xlsx")
    )
    df_hydro["technology"] = "Hydro ROR"
    is_domestic = df_hydro["Country"] == "PL"
    df_hydro.loc[is_domestic, "p_max_pu"] = df_hydro.loc[
        is_domestic, "Hydro capacity utilization"
    ]
    df_hydro.loc[~is_domestic, "p_max_pu_annual"] = df_hydro.loc[
        ~is_domestic, "Hydro capacity utilization"
    ]
    df_hydro = df_hydro.drop(columns=["Hydro capacity utilization"])
    df["Country"] = df["area"].str[:2]
    df = pd.merge(df, df_hydro, on=["Country", "technology"], how="left")
    df = df.drop(columns=["Country"])

    # Convert gross to net power by setting p_max_pu < 1
    # But only for PL - assume capacities for other countries are net
    is_domestic = df["area"].str.startswith("PL")
    df.loc[is_domestic, "p_max_pu"] = (
        df.loc[is_domestic, "p_max_pu"]
        .fillna(df.loc[is_domestic, "Net to gross power ratio"])
        .round(2)
    )

    # The rest has to be filled with 1
    df["p_max_pu"] = df["p_max_pu"].fillna(1)

    # Assume fixed capacity utilization for industrial units
    is_industrial = df["category"] == "Industrial"
    df.loc[is_industrial, "p_max_pu"] *= industrial_utilization
    df.loc[is_industrial, "p_min_pu"] = df.loc[is_industrial, "p_max_pu"]
    df.loc[is_industrial, "p_max_pu_annual"] = 1.0

    # Set remaining parameters

    # If input is of total capacity type, fix the lifetime to 1 year
    df.loc[df["value_type"] == "total", "lifetime"] = 1
    df["lifetime"] = df["lifetime"].fillna(df["Lifetime [years]"])
    df["retire_year"] = df["build_year"] + df["lifetime"]
    if decommission_year_inclusive:
        df["retire_year"] -= 1

    df["efficiency"] = df["Net electrical efficiency"].fillna(1.0)

    # Storage unit specific
    df["max_hours"] = (1 / df["Discharging power [MW/MWh]"]).round(3)
    df["efficiency_round"] = df["Round trip efficiency"]
    sqrt_efficiency = np.round(np.sqrt(df["efficiency_round"]), 3)
    df["efficiency_store"] = sqrt_efficiency
    df["efficiency_dispatch"] = sqrt_efficiency
    df["standing_loss"] = (df["Standing loss per day"] / 24).round(5).fillna(0)

    # Custom parameter p_mean_max_pu = maximum annual capacity utilization factor
    df["Maximum utilization due to outages"] = (
        1
        - df["Planned outage as year fraction"].fillna(0)
        - df["Forced outage as year fraction"].fillna(0)
    )
    df["p_max_pu_annual"] = (
        df["p_max_pu_annual"]
        .fillna(df["p_max_pu"] * df["Maximum utilization due to outages"])
        .round(3)
    )
    # For wind and PV weather is a stronger bound so ignore
    df.loc[
        df["carrier"].str.startswith("Wind") | df["carrier"].str.startswith("PV"),
        "p_max_pu_annual",
    ] = 1
    # For biomass and biogas enforce 80% of the maximum annual capacity utilization factor
    if enforce_bio > 0:
        is_bioenergy = df["category"].str.startswith("Bio")
        df.loc[is_bioenergy, "p_min_pu_annual"] = (
            enforce_bio * df.loc[is_bioenergy, "p_max_pu_annual"]
        ).round(3)

    # Capital cost = annualized investment cost + fixed cost
    # For fixed capacities don't calculate the capital cost
    df["capital_cost"] = 0
    df.loc[df["value_type"] == "addition", "capital_cost"] = (
        1e6
        * df["Investment cost [MPLN/MW_e]"]
        * calculate_annuity(lifetime=df["lifetime"], discount_rate=discount_rate)
        + df["Fixed cost [PLN/MW_e/year]"]
    ).round(3)

    df = df.dropna(axis=1, how="all")

    # By default do not extend capacities
    df["p_nom_extendable"] = False

    # Remove capacities with zero nominal power if they are not to be extended
    if extendable_technologies:
        is_extendable_tech = df["technology"].isin(extendable_technologies)
        df["p_nom_extendable"] = True
        if extend_from_zero:
            df.loc[is_extendable_tech, "p_nom"] = 0
        df["p_nom_min"] = df["p_nom"]
        df["p_nom_max"] = df["p_nom"]
        if active_investment_years:
            is_investment_active = (
                df["build_year"].isin(active_investment_years) & is_extendable_tech
            )
            df.loc[is_investment_active, "p_nom_max"] = np.inf
        df = df[df["p_nom_max"] > 0]
    else:
        df = df[df["p_nom"] > 0]

    return df
