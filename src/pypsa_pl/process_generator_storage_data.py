import pandas as pd
import numpy as np

from pypsa_pl.config import data_dir
from pypsa_pl.io import read_excel
from pypsa_pl.process_technology_data import load_technology_data, get_technology_year
from pypsa_pl.helper_functions import calculate_annuity


def process_utility_units_data(
    source_combustion,
    source_renewable,
    source_storage,
    source_technology,
    decommission_year_inclusive=True,
    warm_reserve_sources=None,
    cold_reserve_sources=None,
):
    """
    Process data on individual utility units (combustion, renewable, storage).
    Long form column names are assumed.
    """

    dfs = []

    for prefix, source in [
        ("combustion", source_combustion),
        ("renewable", source_renewable),
        ("storage", source_storage),
    ]:
        df = read_excel(
            data_dir("input", f"{prefix}_units;source={source}.xlsx"),
            sheet_var="category",
        )

        include_col = "Include in model [YES/NO]"
        if include_col in df.columns:
            df = df[df[include_col] == "YES"].drop(columns=include_col)

        # Mapping of basic data to short form column names
        df = df.rename(
            columns={
                "Unit name": "name",
                "Unit technology": "technology",
                "Main fuel": "carrier",
                "Commissioning date": "build_year",
                "Longitude": "x",
                "Latitude": "y",
            }
        )
        # Use going into reserve as end year if available
        # if "Reserve date" in df.columns:
        #     df["end_year"] = df["Reserve date"].fillna(df["Decommissioning date"])
        # else:
        #     df["end_year"] = df["Decommissioning date"]
        df["end_year"] = df["Decommissioning date"]

        # Carrier equals category for renewables and storage
        if prefix in ["storage", "renewable"]:
            df["carrier"] = df["category"]

        # TODO: add voivodeship information in the input files for renewables and storage
        df["Country"] = "PL"
        # TODO: remove the inconsistency
        if "Voivodeship" in df.columns:
            df["Voivodeship"] = "PL " + df["Voivodeship"]
        df.loc[df["carrier"] == "Wind offshore", "Voivodeship"] = "offshore"

        # Infer technology data
        # Assume the oldest units were virtually built in 2020
        df.loc[(df["build_year"] < 2020) | df["build_year"].isna(), "build_year"] = 2020
        df["end_year"] = df["end_year"].fillna(np.inf)
        df["Lifetime [years]"] = df["end_year"] - df["build_year"]
        if decommission_year_inclusive:
            df["Lifetime [years]"] += 1
        if prefix == "storage":
            df["Discharging power [MW/MWh]"] = (
                df["Gross installed electrical generation capacity [MW_e]"]
                / df["Gross electrical storage capacity [MWh_e]"]
            )

        # Match the build year to technology year for 5 year investment periods
        # e.g. 2020 -> 2015, 2023 -> 2020, 2025 -> 2020, 2026 -> 2025
        df["technology_year"] = get_technology_year(df["build_year"]).astype(int)

        # Fill missing parameters by default technology data
        technology_years = df["technology_year"].drop_duplicates().tolist()
        df_tech = load_technology_data(source=source_technology, years=technology_years)
        param_cols = [
            col for col in df_tech.columns if col not in ["year", "technology"]
        ]
        df_tech = df_tech.rename(columns={"year": "technology_year"})
        df = (
            df.set_index("name")
            .combine_first(
                pd.merge(
                    df.drop(columns=[col for col in param_cols if col in df.columns]),
                    df_tech,
                    on=["technology_year", "technology"],
                    how="left",
                ).set_index("name"),
            )
            .reset_index()
        )

        # Change names to PyPSA convention
        df["p_nom"] = df["Gross installed electrical generation capacity [MW_e]"].round(
            3
        )
        df["lifetime"] = df["Lifetime [years]"]

        df["p_max_pu"] = df["Net to gross power ratio"].round(2).fillna(1)
        # Custom parameter p_mean_max_pu = maximum annual capacity utilization factor
        df["p_max_pu_annual"] = (
            df["p_max_pu"]
            * (
                1
                - df["Planned outage as year fraction"].fillna(0)
                - df["Forced outage as year fraction"].fillna(0)
            )
        ).round(3)
        # For wind and PV weather is a stronger bound so ignore
        df.loc[
            df["carrier"].str.startswith("Wind") | df["carrier"].str.startswith("PV"),
            "p_max_pu_annual",
        ] = 1

        # Combustion unit assumptions
        if prefix == "combustion":
            df["efficiency"] = df["Net electrical efficiency"]

        # Renewable unit assumptions
        if prefix == "renewable":
            df["efficiency"] = 1.0

        # Storage unit assumptions
        if prefix == "storage":
            df["max_hours"] = (1 / df["Discharging power [MW/MWh]"]).round(3)
            sqrt_efficiency = np.round(np.sqrt(df["Round trip efficiency"]), 3)
            df["efficiency_store"] = sqrt_efficiency
            df["efficiency_dispatch"] = sqrt_efficiency
            df["standing_loss"] = (df["Standing loss per day"] / 24).round(5).fillna(0)
            if "Average power inflow [MW_e]" in df.columns:
                df["inflow"] = df["Average power inflow [MW_e]"].fillna(0)
            if "Charging capacity [MW_e]" in df.columns:
                df["p_min_pu"] = (-1) * np.minimum(
                    (df["Charging capacity [MW_e]"] / df["p_nom"]).round(3), 1
                )

        # For fixed units don't calculate the capital cost
        df["capital_cost"] = 0

        # Capacity of individual units is not to be extended
        df["p_nom_extendable"] = False

        # Contribution to reserve
        df["is_warm_reserve"] = False
        if warm_reserve_sources:
            df.loc[df["category"].isin(warm_reserve_sources), "is_warm_reserve"] = True
        df["is_cold_reserve"] = False
        if cold_reserve_sources:
            df.loc[df["category"].isin(cold_reserve_sources), "is_cold_reserve"] = True

        df = df.dropna(axis=1, how="all")
        dfs.append(df)

    df = pd.concat(dfs)
    return df


def process_aggregate_capacity_data(
    df,
    area_column="Voivodeship",
    source_technology="instrat_2022",
    source_hydro_utilization="entsoe_2020",
    discount_rate=0.05,
    extendable_technologies=None,
    warm_reserve_sources=None,
    cold_reserve_sources=None,
    active_investment_years=None,
    extend_from_zero=False,
    enforce_bio=0,
    industrial_utilization=0.5,
):
    """
    Process data on aggregate generation and storage capacities by year and by area (voivodeship or country).
    """

    # if input is of total capacity type, fix the lifetime to 1 year
    df.loc[df["type"] == "capacity", "lifetime"] = 1

    df = df[["year", "type", area_column, "category", "p_nom", "lifetime"]].copy()

    # Assumptions
    # TODO: simplify this part
    assumptions = {
        "PV ground": {
            "carrier": "PV ground",
            "technology": "PV ground",
        },
        "PV roof": {
            "carrier": "PV roof",
            "technology": "PV roof",
        },
        "Wind onshore": {
            "carrier": "Wind onshore",
            "technology": "Wind onshore",
        },
        "Wind offshore": {
            "carrier": "Wind offshore",
            "technology": "Wind offshore",
        },
        "Hydro ROR": {
            "carrier": "Hydro ROR",
            "technology": "Hydro ROR",
        },
        "Biogas": {
            "carrier": "Biogas",
            "technology": "Biogas",
        },
        "Biomass straw": {
            "carrier": "Biomass straw",
            "technology": "Biomass straw",
        },
        "Natural gas CCGT": {
            "carrier": "Natural gas",
            "technology": "Natural gas CCGT",
        },
        "Natural gas OCGT": {
            "carrier": "Natural gas",
            "technology": "Natural gas OCGT",
        },
        "Hard coal": {
            "carrier": "Hard coal",
            "technology": "Hard coal",
        },
        "Lignite": {
            "carrier": "Lignite",
            "technology": "Lignite",
        },
        "Biomass wood": {
            "carrier": "Biomass wood chips",
            "technology": "Biomass wood chips",
        },
        "Battery large 4h": {
            "carrier": "Battery large",
            "technology": "Battery large 4h",
        },
        "Battery small 2h": {
            "carrier": "Battery small",
            "technology": "Battery small 2h",
        },
        # Technologies which are always defined by total capacities and not by investments
        "Biomass straw old": {
            "carrier": "Biomass straw",
            "technology": "Biomass straw",
        },
        "Wind onshore old": {
            "carrier": "Wind onshore",
            "technology": "Wind onshore",
        },
        "Natural gas industrial": {
            "carrier": "Natural gas",
            "technology": "Natural gas CCGT",
        },
        "Hard coal industrial": {
            "carrier": "Hard coal",
            "technology": "Hard coal",
        },
        "DSR": {
            "carrier": "DSR",
            "technology": "DSR",
        },
        # Technologies relevant for foreign nodes
        "Nuclear": {
            "carrier": "Nuclear",
            "technology": "Nuclear",
        },
        "Oil": {
            "carrier": "Oil",
            "technology": "Oil",
        },
        "PV": {
            "carrier": "PV",
            "technology": "PV",
        },
        "Hydro PSH": {
            "carrier": "Hydro PSH",
            "technology": "Hydro PSH",
        },
    }
    for category, attributes in assumptions.items():
        for key, value in attributes.items():
            df.loc[df["category"] == category, key] = value

    df["p_nom"] = df["p_nom"].round(3)
    # Load technological assumptions
    df["technology_year"] = get_technology_year(df["year"]).astype(int)
    technology_years = df["technology_year"].drop_duplicates().tolist()
    df_tech = load_technology_data(source=source_technology, years=technology_years)
    param_cols = [col for col in df_tech.columns if col not in ["year", "technology"]]
    df_tech = df_tech.rename(columns={"year": "technology_year"})
    df = (
        df.set_index([area_column, "category", "year"])
        .combine_first(
            pd.merge(
                df.drop(columns=[col for col in param_cols if col in df.columns]),
                df_tech,
                on=["technology_year", "technology"],
                how="left",
            ).set_index([area_column, "category", "year"]),
        )
        .reset_index()
    )

    # Load hydro utilization parameters as p_max_pu_annual
    df_hydro = read_excel(
        data_dir("input", f"hydro_utilization;source={source_hydro_utilization}.xlsx")
    )
    df_hydro["technology"] = "Hydro ROR"

    if area_column == "Country":
        df_hydro = df_hydro.rename(
            columns={"Hydro capacity utilization": "p_max_pu_annual"}
        )
        df = pd.merge(df, df_hydro, on=["Country", "technology"], how="left")
    else:
        df_hydro = df_hydro.rename(columns={"Hydro capacity utilization": "p_max_pu"})
        df_hydro = df_hydro[df_hydro["Country"] == "PL"].drop(columns=["Country"])
        df = pd.merge(df, df_hydro, on=["technology"], how="left")
        df["p_max_pu_annual"] = np.nan

    # Convert gross to net power by setting p_max_pu < 1
    # But only for PL - assume capacities for other countries are net
    if area_column == "Voivodeship":
        if "p_max_pu" in df.columns:
            df["p_max_pu"] = (
                df["p_max_pu"].fillna(df["Net to gross power ratio"]).round(2).fillna(1)
            )
        else:
            df["p_max_pu"] = df["Net to gross power ratio"].round(2).fillna(1)

    else:
        if "p_max_pu" in df.columns:
            df["p_max_pu"] = df["p_max_pu"].fillna(1)
        else:
            df["p_max_pu"] = 1

    # Assume fixed capacity utilization for industrial units
    is_industrial = df["category"].str.contains("industrial")
    df.loc[is_industrial, "p_max_pu"] *= industrial_utilization
    df.loc[is_industrial, "p_min_pu"] = df.loc[is_industrial, "p_max_pu"]
    df.loc[is_industrial, "p_max_pu_annual"] = 1.0

    # Define PyPSA parameters
    df = df.rename(columns={"year": "build_year"})
    df["lifetime"] = df["lifetime"].fillna(df["Lifetime [years]"])
    df["efficiency"] = df["Net electrical efficiency"].fillna(1.0)
    df["max_hours"] = (1 / df["Discharging power [MW/MWh]"]).round(3)
    sqrt_efficiency = np.round(np.sqrt(df["Round trip efficiency"]), 3)
    df["efficiency_store"] = sqrt_efficiency
    df["efficiency_dispatch"] = sqrt_efficiency
    df["standing_loss"] = (df["Standing loss per day"] / 24).round(5).fillna(0)
    if "Average power inflow [MW_e]" in df.columns:
        df["inflow"] = df["Average power inflow [MW_e]"].fillna(0)
    if "Charging capacity [MW_e]" in df.columns:
        df["p_min_pu"] = (-1) * np.minimum(
            (df["Charging capacity [MW_e]"] / df["p_nom"]).round(3), 1
        )
    # Custom parameter p_mean_max_pu = maximum annual capacity utilization factor
    df["p_max_pu_annual"] = (
        df["p_max_pu_annual"]
        .fillna(
            df["p_max_pu"]
            * (
                1
                - df["Planned outage as year fraction"].fillna(0)
                - df["Forced outage as year fraction"].fillna(0)
            )
        )
        .round(3)
    )
    # For wind and PV weather is a stronger bound so ignore
    df.loc[
        df["carrier"].str.startswith("Wind") | df["carrier"].str.startswith("PV"),
        "p_max_pu_annual",
    ] = 1
    # For biomass and biogas enforce 80% of the maximum annual capacity utilization factor
    if enforce_bio > 0:
        is_bioenergy = df["carrier"].str.startswith("Bio")
        df.loc[is_bioenergy, "p_min_pu_annual"] = (
            enforce_bio * df.loc[is_bioenergy, "p_max_pu_annual"]
        ).round(3)

    # Capital cost = annualized investment cost + fixed cost
    # For fixed capacities don't calculate the capital cost
    df["capital_cost"] = 0
    df.loc[df["type"] == "investment", "capital_cost"] = (
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
        is_extendable_tech = df["category"].isin(extendable_technologies)
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

    df["is_warm_reserve"] = False
    if area_column == "Voivodeship" and warm_reserve_sources:
        df.loc[df["category"].isin(warm_reserve_sources), "is_warm_reserve"] = True
    df["is_cold_reserve"] = False
    if area_column == "Voivodeship" and cold_reserve_sources:
        df.loc[df["category"].isin(cold_reserve_sources), "is_cold_reserve"] = True

    return df
