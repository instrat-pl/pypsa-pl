import pandas as pd
import numpy as np

from pypsa_pl.helper_functions import select_subset, repeat_over_timesteps


def add_generators(network, df_generators, dfs_capacity_factors, df_srmc):
    carriers = [
        "Wind onshore",
        "Wind offshore",
        "PV roof",
        "PV ground",
        "DSR",
        "Hard coal",
        "Natural gas",
        "Lignite",
        "Biogas",
        "Biomass wood chips",
        "Biomass straw",
        "Coke-oven gas",
        "Oil",
        "Nuclear",
        "Hydro ROR",
    ]
    df_generators = select_subset(df_generators, "carrier", carriers)

    for category, df in df_generators.groupby("category"):
        names = df["name"].tolist()
        df = df.set_index("name")
        attributes = [
            "bus",
            "area",
            "category",
            "carrier",
            "technology",
            "p_nom",
            "p_nom_min",
            "p_nom_max",
            "p_min_pu",
            "p_max_pu",
            "p_set",
            "p_min_pu_annual",
            "p_max_pu_annual",
            "efficiency",
            "marginal_cost",
            "capital_cost",
            "technology_year",
            "build_year",
            "retire_year",
            "lifetime",
            "p_nom_extendable",
            "is_warm_reserve",
            "is_cold_reserve",
        ]
        kwargs = {key: df[key] for key in attributes if key in df.columns}
        for attribute, default in {
            "p_min_pu": 0.0,
            "p_max_pu": 1.0,
            "p_min_pu_annual": 0.0,
            "p_max_pu_annual": 1.0,
        }.items():
            if attribute in kwargs.keys():
                kwargs[attribute] = kwargs[attribute].fillna(default)
            else:
                kwargs[attribute] = default

        if category.startswith("CHP"):
            # CHP units are considered only for PL
            assert df["area"].str.startswith("PL").all()

            df_cf = dfs_capacity_factors[category].set_index(["period", "timestep"])

            areas = df["area"]
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
            p_max = df.loc[names, ["p_nom", "p_max_pu"]].product(axis=1).transpose()
            kwargs["p_set"] = (p_max * df_cf).round(3)

        if category.startswith("Wind") or category.startswith("PV"):
            df_cf = dfs_capacity_factors[category].set_index(["period", "timestep"])

            areas = df["area"]
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
            if len(df_srmc_cat["year"].unique()) > 1:
                df_srmc_cat = repeat_over_timesteps(df_srmc_cat, network).set_index(
                    ["period", "timestep"]
                )
            else:
                df_srmc_cat = df_srmc_cat.drop(columns="year").iloc[0, :]
            kwargs["marginal_cost"] = df_srmc_cat

        kwargs = {key: kwargs[key] for key in attributes if key in kwargs.keys()}

        network.madd(
            "Generator",
            names,
            **kwargs,
        )


def add_storage(network, df_storage_units, df_capacity_factors, df_srmc):
    carriers = [
        "Hydro PSH",
        "Battery large",
        "Battery small",
    ]
    df_storage_units = select_subset(df_storage_units, "carrier", carriers)

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
            "efficiency_round",
            "efficiency_store",
            "efficiency_dispatch",
            "inflow",
            "standing_loss",
            "cyclic_state_of_charge",
            "marginal_cost",
            "capital_cost",
            "technology_year",
            "build_year",
            "retire_year",
            "lifetime",
            "p_nom_extendable",
            "is_warm_reserve",
            "is_cold_reserve",
        ]
        kwargs = {key: df[key] for key in attributes if key in df.columns}
        for attribute, default in {
            "p_min_pu": -1.0,
            "inflow": 0.0,
            "cyclic_state_of_charge": True,
        }.items():
            if attribute in kwargs.keys():
                kwargs[attribute] = kwargs[attribute].fillna(default)
            else:
                kwargs[attribute] = default

        srmc_names = [name for name in names if name in df_srmc.columns]
        if len(srmc_names) > 0:
            df_srmc_cat = df_srmc[["year", *srmc_names]]
            if len(df_srmc_cat["year"].unique()) > 1:
                df_srmc_cat = repeat_over_timesteps(df_srmc_cat, network).set_index(
                    ["period", "timestep"]
                )
            else:
                df_srmc_cat = df_srmc_cat.drop(columns="year").iloc[0, :]
            kwargs["marginal_cost"] = df_srmc_cat

        kwargs = {key: kwargs[key] for key in attributes if key in kwargs.keys()}

        network.madd(
            "StorageUnit",
            names,
            **kwargs,
        )
