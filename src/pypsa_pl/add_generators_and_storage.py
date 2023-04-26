import pandas as pd
import numpy as np

from pypsa_pl.helper_functions import select_subset, repeat_over_timesteps


def add_generators(network, df_generators, dfs_capacity_factors, df_srmc):
    categories = [
        "JWCD",
        "nJWCD",
        "CHP",
        "Wind onshore",
        "Wind onshore old",
        "Wind offshore",
        "PV roof",
        "PV ground",
        "PV",
        "DSR",
        "Hard coal industrial",
        "Natural gas industrial",
        "Biogas",
        "Biomass wood",
        "Biomass straw",
        "Biomass straw old",
        "Hard coal",
        "Lignite",
        "Natural gas OCGT",
        "Natural gas CCGT",
        "Oil",
        "Nuclear",
        "Hydro ROR",
    ]
    df_generators = select_subset(df_generators, "category", categories)

    for cat, df in df_generators.groupby("category"):
        names = df["name"].tolist()
        df = df.set_index("name")
        attributes = [
            "bus",
            "carrier",
            "p_nom",
            "p_nom_min",
            "p_nom_max",
            "p_min_pu",
            "p_max_pu",
            "p_min_pu_annual",
            "p_max_pu_annual",
            "efficiency",
            "marginal_cost",
            "capital_cost",
            "build_year",
            "lifetime",
            "p_nom_extendable",
            "is_warm_reserve",
            "is_cold_reserve",
        ]
        kwargs = {key: df[key] for key in attributes if key in df.columns}
        if "p_max_pu" in kwargs.keys():
            kwargs["p_max_pu"] = kwargs["p_max_pu"].fillna(1.0)
        if "p_min_pu" in kwargs.keys():
            kwargs["p_min_pu"] = kwargs["p_min_pu"].fillna(0.0)

        # TODO: cleaner duplication of columns
        if cat == "CHP":  # or cat == "JWCD":
            df_cf = dfs_capacity_factors["CHP"].set_index(["period", "timestep"])

            # CHP units are considered only for PL
            assert (df["Country"] == "PL").all()
            names = df.index
            # if cat == "JWCD":
            #     names = df.index[df.index.str.startswith("EC")]

            areas_fuels = df[["Voivodeship", "carrier"]].copy()
            areas_fuels.loc[
                areas_fuels["carrier"].isin(
                    ["Hard coal", "Lignite", "Biomass wood chips"]
                ),
                "carrier",
            ] = "Coal"
            generators_per_area_fuel = areas_fuels.groupby(
                ["Voivodeship", "carrier"]
            ).groups

            assert set(generators_per_area_fuel.keys()).issubset(df_cf.columns)
            df_cf = pd.concat(
                [
                    df_cf[[area_fuel]].rename(columns={area_fuel: generator})
                    for area_fuel, generators in generators_per_area_fuel.items()
                    for generator in generators
                ],
                axis=1,
            )
            p_max = df.loc[names, ["p_nom", "p_max_pu"]].product(axis=1).transpose()
            kwargs["p_set"] = (p_max * df_cf).round(3)

        if cat.startswith("Wind") or cat.startswith("PV"):
            if cat.startswith("PV"):
                key = "PV"
            # elif cat == "Wind onshore old" and cat not in dfs_capacity_factors.keys():
            #     key = "Wind onshore"
            else:
                key = cat
            df_cf = dfs_capacity_factors[key].set_index(["period", "timestep"])

            areas = df["Country"].copy()
            if "Voivodeship" in df.columns:
                has_voivodeship = df["Voivodeship"].notna()
                areas.loc[has_voivodeship] += (
                    " " + df.loc[has_voivodeship, "Voivodeship"]
                )
            generators_per_area = areas.groupby(areas).groups
            assert set(generators_per_area.keys()).issubset(df_cf.columns)
            df_cf = pd.concat(
                [
                    df_cf[[area]].rename(columns={area: generator})
                    for area, generators in generators_per_area.items()
                    for generator in generators
                ],
                axis=1,
            )
            kwargs["p_max_pu"] = df_cf

        srmc_names = [name for name in names if name in df_srmc.columns]
        if len(srmc_names) > 0:
            df_srmc_cat = df_srmc[["year", *srmc_names]]
            df_srmc_cat = repeat_over_timesteps(df_srmc_cat, network).set_index(
                ["period", "timestep"]
            )
            kwargs["marginal_cost"] = df_srmc_cat

        network.madd(
            "Generator",
            names,
            suffix=f" {cat}",
            **kwargs,
        )


def add_storage(network, df_storage_units, df_capacity_factors, df_srmc):
    categories = [
        "Hydro PSH",
        "Battery large",
        "Battery large 1h",
        "Battery large 4h",
        "Battery small 2h",
    ]
    df_storage_units = select_subset(df_storage_units, "category", categories)

    for cat, df in df_storage_units.groupby("category"):
        names = df["name"].tolist()
        df = df.set_index("name")
        attributes = [
            "bus",
            "carrier",
            "p_nom",
            "p_nom_min",
            "p_nom_max",
            "p_min_pu",
            # "p_max_pu",
            # "p_min_pu_annual",
            # "p_max_pu_annual",
            "max_hours",
            "efficiency_store",
            "efficiency_dispatch",
            "inflow",
            "standing_loss",
            "marginal_cost",
            "capital_cost",
            "build_year",
            "lifetime",
            "p_nom_extendable",
            "is_warm_reserve",
            "is_cold_reserve",
        ]
        kwargs = {key: df[key] for key in attributes if key in df.columns}
        if "p_min_pu" in kwargs.keys():
            kwargs["p_min_pu"] = kwargs["p_min_pu"].fillna(-1.0)
        if "inflow" in kwargs.keys():
            kwargs["inflow"] = kwargs["inflow"].fillna(0.0)

        srmc_names = [name for name in names if name in df_srmc.columns]
        if len(srmc_names) > 0:
            df_srmc_cat = df_srmc[["year", *srmc_names]]
            df_srmc_cat = repeat_over_timesteps(df_srmc_cat, network).set_index(
                ["period", "timestep"]
            )
            kwargs["marginal_cost"] = df_srmc_cat

        network.madd(
            "StorageUnit",
            names,
            suffix=f" {cat}",
            cyclic_state_of_charge=True,
            **kwargs,
        )
