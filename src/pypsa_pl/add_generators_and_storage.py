import pandas as pd
import numpy as np

from pypsa_pl.helper_functions import select_subset, repeat_over_timesteps


def add_generators(network, df_generators, dfs_capacity_factors, df_srmc, fix_chp=True):
    carriers = [
        "Wind onshore",
        "Wind offshore",
        "PV roof",
        "PV ground",
        "DSR",
        "Hard coal",
        "Natural gas",
        "Lignite",
        "Biogas",  # biogas plant and engine only
        "Biomass wood chips",
        "Biomass straw",
        "Coke-oven gas",
        "Oil",
        "Nuclear large",
        "Nuclear small",
        "Hydro ROR",
        "Biogas plant",
        # Sectoral
        "ICE vehicle",
        "Natural gas reforming",
        "Conventional heating plant",
        "Conventional boiler",
        "Biomass boiler",
        # CHPs
        "Hard coal heat output",
        "Natural gas heat output",
        "Lignite heat output",
        "Biogas heat output",
        "Biomass wood chips heat output",
        "Biomass straw heat output",
        "Hydrogen heat output",
    ]
    df_generators = select_subset(df_generators, "carrier", carriers)
    # TODO: select generators not based on carrier but on technology
    df_generators = df_generators[
        ~df_generators["technology"].isin(["Biogas engine", "Biogas engine CHP"])
    ]

    for category, df in df_generators.groupby("category"):
        names = df["name"].tolist()
        df = df.set_index("name")
        attributes = [
            "bus",
            "area",
            "sector",
            "category",
            "carrier",
            "technology",
            "technology_bundle",
            "p_nom",
            "p_nom_min",
            "p_nom_max",
            "p_min_pu",
            "p_min_pu_stable",
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
            "committable",
            "start_up_cost",
            "shut_down_cost",
            "min_up_time",
            "min_down_time",
            "ramp_limit_start_up",
            "ramp_limit_shut_down",
            "ramp_limit_up",
            "ramp_limit_down",
            "is_primary_reserve",
            "is_tertiary_reserve",
            "is_cold_reserve",
            "primary_reserve_ramp_limit_up",
            "primary_reserve_ramp_limit_down",
            "tertiary_reserve_ramp_limit_up",
        ]
        kwargs = {key: df[key] for key in attributes if key in df.columns}
        for attribute, default in {
            "p_min_pu": 0.0,
            "p_max_pu": 1.0,
            "p_min_pu_annual": 0.0,
            "p_max_pu_annual": 1.0,
            "min_up_time": 0,
            "min_down_time": 0,
            "committable": False,
        }.items():
            if attribute in kwargs.keys():
                kwargs[attribute] = kwargs[attribute].fillna(default)
            else:
                kwargs[attribute] = default

        if category.startswith("CHP") and fix_chp:
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


def add_links(network, df_links, dfs_capacity_factors, df_srmc):
    # Generators that use synthetic fuels (e.g. hydrogen)
    # Energy form conversion (e.g. electricity to hydrogen)
    # BEV charger / engine / V2G
    carriers = [
        "Biogas",  # Biogas engine only
        "Hydrogen",
        "Electrolyser",
        "Heat pump small",
        "Heat pump large",
        "Resistive heater large",
        "Resistive heater small",
        "BEV",
        "BEV charger",
        "BEV V2G",
    ]
    df_links = select_subset(df_links, "carrier", carriers)
    # TODO: select links not based on carrier but on technology
    df_links = df_links[
        ~df_links["technology"].str.startswith("Biogas plant and engine")
    ]

    for category, df in df_links.groupby("category"):
        names = df["name"].tolist()
        df = df.set_index("name")
        attributes = [
            "bus0",
            "bus1",
            "area",
            "sector",
            "category",
            "carrier",
            "technology",
            "technology_bundle",
            "p_nom",
            "p_nom_min",
            "p_nom_max",
            "p0_sign",
            "p_min_pu",
            "p_min_pu_stable",
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
            "committable",
            "start_up_cost",
            "shut_down_cost",
            "min_up_time",
            "min_down_time",
            "ramp_limit_start_up",
            "ramp_limit_shut_down",
            "ramp_limit_up",
            "ramp_limit_down",
            "is_primary_reserve",
            "is_tertiary_reserve",
            "is_cold_reserve",
            "primary_reserve_ramp_limit_up",
            "primary_reserve_ramp_limit_down",
            "tertiary_reserve_ramp_limit_up",
        ]
        kwargs = {key: df[key] for key in attributes if key in df.columns}
        for attribute, default in {
            "p0_sign": 1,
            "p_min_pu": 0.0,
            "p_max_pu": 1.0,
            "p_min_pu_annual": 0.0,
            "p_max_pu_annual": 1.0,
            "min_up_time": 0,
            "min_down_time": 0,
            "committable": False,
        }.items():
            if attribute in kwargs.keys():
                kwargs[attribute] = kwargs[attribute].fillna(default)
            else:
                kwargs[attribute] = default

        if isinstance(kwargs["p0_sign"], pd.Series):
            negative_sign = kwargs["p0_sign"][kwargs["p0_sign"] < 0].index
        else:
            negative_sign = []

        if category in ["BEV charger", "BEV V2G"]:
            df_cf = dfs_capacity_factors[category].set_index(["period", "timestep"])

            areas = df["area"]
            links_per_area = areas.groupby(areas).groups
            assert set(links_per_area.keys()).issubset(df_cf.columns)
            df_cf = pd.concat(
                [
                    df_cf[[area]].rename(columns={area: link})
                    for area, links in links_per_area.items()
                    for link in links
                ],
                axis=1,
            )
            kwargs["p_max_pu"] = df_cf[df_cf.columns.difference(negative_sign)]
            if len(negative_sign) > 0:
                kwargs["p_min_pu"] = -df_cf[negative_sign]

        if category.startswith("Heat pump"):
            # TODO: generalize dfs_capacity_factors to general timeseries
            df_eff = dfs_capacity_factors[category].set_index(["period", "timestep"])

            areas = df["area"]
            links_per_area = areas.groupby(areas).groups
            assert set(links_per_area.keys()).issubset(df_eff.columns)
            df_eff = pd.concat(
                [
                    df_eff[[area]].rename(columns={area: link})
                    for area, links in links_per_area.items()
                    for link in links
                ],
                axis=1,
            )
            df_eff[negative_sign] = (1 / df_eff[negative_sign]).round(4)
            kwargs["efficiency"] = df_eff

        srmc_names = [name for name in names if name in df_srmc.columns]
        if len(srmc_names) > 0:
            df_srmc_cat = df_srmc[["year", *srmc_names]]
            if len(df_srmc_cat["year"].unique()) > 1:
                df_srmc_cat = repeat_over_timesteps(df_srmc_cat, network).set_index(
                    ["period", "timestep"]
                )
            else:
                df_srmc_cat = df_srmc_cat.drop(columns="year").iloc[0, :]
            df_srmc_cat[negative_sign] *= -1
            kwargs["marginal_cost"] = df_srmc_cat

        kwargs = {key: kwargs[key] for key in attributes if key in kwargs.keys()}

        network.madd(
            "Link",
            names,
            **kwargs,
        )


def add_storage_units(network, df_storage_units, df_capacity_factors, df_srmc):
    # Storage units are defined by both their energy storage capacity and charge/discharge power
    # Ratio of energy storage capacity to charge/discharge power is fixed by max_hours
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
            "area",
            "sector",
            "category",
            "carrier",
            "technology",
            "technology_bundle",
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
            "is_primary_reserve",
            "is_tertiary_reserve",
            "is_cold_reserve",
            "primary_reserve_ramp_limit_up",
            "primary_reserve_ramp_limit_down",
            "tertiary_reserve_ramp_limit_up",
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


def add_stores(network, df_stores, df_capacity_factors, df_srmc):
    # Stores are defined by energy storage capacity only
    carriers = [
        "Biogas storage",
        "Hydrogen storage",
        "Heat storage small",
        "Heat storage large",
        "BEV battery",
    ]
    df_stores = select_subset(df_stores, "carrier", carriers)

    for cat, df in df_stores.groupby("category"):
        names = df["name"].tolist()
        df = df.set_index("name")
        attributes = [
            "bus",
            "area",
            "sector",
            "category",
            "carrier",
            "technology",
            "technology_bundle",
            "e_nom",
            "e_nom_min",
            "e_nom_max",
            "e_nom_extendable",
            "standing_loss",
            "e_cyclic",
            "marginal_cost",
            "capital_cost",
            "technology_year",
            "build_year",
            "retire_year",
            "lifetime",
        ]
        kwargs = {key: df[key] for key in attributes if key in df.columns}
        for attribute, default in {
            "e_cyclic": True,
            "e_nom_min": 0,
            "e_nom_max": np.inf,
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
            "Store",
            names,
            **kwargs,
        )
