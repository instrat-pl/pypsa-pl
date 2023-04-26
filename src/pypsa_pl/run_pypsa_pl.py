import json
from math import inf
import pandas as pd
import numpy as np
import os
import logging
from contextlib import contextmanager

import pypsa
from pypsa.networkclustering import get_clustering_from_busmap

from pypsa_pl.config import data_dir
from pypsa_pl.io import read_excel
from pypsa_pl.helper_functions import select_subset, repeat_over_periods
from pypsa_pl.make_network import make_network
from pypsa_pl.process_capacity_factors_data import process_utilization_profiles
from pypsa_pl.process_generator_storage_data import (
    process_utility_units_data,
    process_aggregate_capacity_data,
)
from pypsa_pl.process_srmc_data import process_srmc_data
from pypsa_pl.add_generators_and_storage import add_generators, add_storage
from pypsa_pl.custom_constraints import (
    p_set_constraint,
    maximum_annual_capacity_factor,
    minimum_annual_capacity_factor,
    warm_reserve,
    cold_reserve,
    maximum_capacity_per_voivodeship,
    maximum_growth_per_carrier,
)


class Params:
    def __init__(self, **kwargs):
        self.set_defaults()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter {key}")

    def __repr__(self):
        return json.dumps(vars(self))

    def set_defaults(self):
        self.temporal_resolution = "1H"
        self.grid_resolution = "copper_plate"
        self.years = [2020]
        self.imports = True
        self.exports = True
        self.trade_factor = 1
        self.dynamic_trade_prices = True
        self.trade_prices = "pypsa_pl_v1"
        self.nodes = "pypsa_pl_v1_75"
        self.interconnectors = "pypsa_pl_v2.1"
        self.default_v_nom = 380
        self.default_build_year = 2020
        self.demand = "PSE_2022"
        self.demand_correction = 1.0
        self.srmc_only_JWCD = False
        self.load_profile = "entsoe"
        self.load_profile_year = 2012
        self.neighbors_load_profile = "PECD3"
        self.neighbors_capacity_demand = "TYNDP_2022"
        self.sectors = ["electricity"]
        self.combustion_units = "energy.instrat.pl"
        self.renewable_units = "pypsa_pl_v2"
        self.storage_units = "pypsa_pl_v2"
        self.capacity_investments = "pypsa_pl_v2_copper_plate"
        self.capacity_potentials = "instrat_2021"
        self.capacity_max_growth = "instrat_2022"
        self.technology_data = "instrat_2023"
        self.hydro_utilization = "entsoe_2020"
        self.renewable_utilization_profiles = "PECD3+EMHIRES"
        self.chp_utilization_profiles = "regression"
        self.prices = "instrat_2023"
        self.scenario = "pypsa_pl_v2"
        self.solver = "highs"
        self.mode = "lopf"
        self.decommission_year_inclusive = True
        self.srmc_wind = 8.0
        self.srmc_pv = 1.0
        self.srmc_dsr = 1200.0
        self.enforce_bio = 0
        self.industrial_utilization = 0.5
        self.correction_factor_wind_old = 0.91
        self.correction_factor_wind_new = 1.09
        self.discount_rate = 0.03
        self.extendable_technologies = None
        self.extend_from_zero = False
        self.warm_reserve_sources = [
            "JWCD",
            "Hydro PSH",
            "Battery large",
            "Battery large 1h",
            "Battery large 4h",
        ]
        self.cold_reserve_sources = ["JWCD"]
        self.warm_reserve_need_per_demand = 0.09
        self.warm_reserve_need_per_pv = 0.15
        self.warm_reserve_need_per_wind = 0.10
        self.cold_reserve_need_per_demand = 0.09
        self.cold_reserve_need_per_import = 1.0
        self.max_r_over_p = 1.0
        self.random_seed = 0
        self.extension_years = 5
        self.virtual_dsr = True


def input_dir(*path):
    return data_dir("input", *path)


def runs_dir(*path):
    return data_dir("runs", *path)


def adjust_lifetime_to_periods(df, years):
    df_lifetime = pd.DataFrame({"year": years})
    df_lifetime["lifetime"] = df_lifetime["year"].diff().shift(-1).fillna(inf)
    if "lifetime" in df.columns:
        df = df.drop(columns="lifetime")
    df = pd.merge(df, df_lifetime, on="year", how="left")
    df["name"] = df["name"] + " " + df["year"].astype(str)
    df = df.rename(columns={"year": "build_year"})
    return df


def split_into_imports_exports(df):
    df_partners = df["name"].str[:5].str.split("_", expand=True)
    df_partners.columns = ["exporter", "importer"]
    df_imports = df[df_partners["importer"] == "PL"].copy()
    df_exports = df[df_partners["exporter"] == "PL"].copy()
    return df_imports, df_exports


@contextmanager
def create_fictional_line(network, busmap):
    network.add("Bus", "fictional")
    network.add("Line", "fictional", bus0="fictional", bus1=busmap.index[0])
    busmap.loc["fictional"] = "fictional"
    yield
    busmap.drop(labels="fictional", inplace=True)
    network.remove("Line", "fictional")
    network.remove("Bus", "fictional")


def calculate_load_timeseries(df_load, df_demand, network, temporal_resolution):
    # Aggregate the load profile to the temporal resolution of the snapshots
    df_load = (
        df_load.groupby(pd.Grouper(key="hour", freq=temporal_resolution))
        .sum()
        .reset_index()
    )
    df_load = df_load.rename(columns={"hour": "timestep"})

    # Merge snapshots and load profile on month, day, and hour
    df_load = repeat_over_periods(df_load, network)

    # Merge snapshots and demand on year
    df_load = df_load.melt(
        id_vars=["period", "timestep"], var_name="country", value_name="load_profile"
    )

    df_demand = df_demand.melt(
        id_vars="year", var_name="country", value_name="annual_demand"
    )

    df_load = pd.merge(
        df_load,
        df_demand,
        left_on=["period", "country"],
        right_on=["year", "country"],
        how="left",
    ).drop(columns="year")
    assert not df_load.isna().any().any()

    # Calculate demand and p_set
    df_load["demand"] = df_load["load_profile"] * df_load["annual_demand"]
    # Demand is in TWh, p_set is in MW
    hours_per_timestep = int(temporal_resolution[:-1])  # e.g. 1H -> 1
    df_load["p_set"] = df_load["demand"] / hours_per_timestep * 1e6

    # Wide format
    df_load = (
        df_load.pivot(index=["period", "timestep"], columns="country", values="p_set")
        .round(3)
        .reset_index()
    )
    return df_load


def add_virtual_dsr(network, srmc_dsr):
    max_load_per_bus = (
        pd.merge(
            network.loads_t["p_set"].reset_index(drop=True).transpose(),
            network.loads[["bus"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        .groupby("bus")
        .sum()
        .max(axis=1)
    )
    network.madd(
        "Generator",
        max_load_per_bus.index,
        suffix=f" Virtual DSR",
        bus=max_load_per_bus.index,
        p_nom=max_load_per_bus,
        carrier="DSR",
        marginal_cost=srmc_dsr,
    )


def run_pypsa_pl(params=Params(), use_cache=False, dry=False):
    logging.info(f"Running PyPSA-PL for parameters: {params}")

    if not use_cache:
        os.makedirs(runs_dir(params.scenario, "input"), exist_ok=True)
        srmc_file = runs_dir(params.scenario, "input", "srmc.csv")
        srmc_file.unlink(missing_ok=True)

        network = make_network(
            temporal_resolution=params.temporal_resolution,
            years=params.years,
            discount_rate=params.discount_rate,
        )
        logging.info("Created the network")

        # Domestic nodes and lines
        df_nodes = read_excel(input_dir(f"nodes;source={params.nodes}.xlsx"))
        df_nodes["name"] = "PL_" + df_nodes["name"].astype(str)
        if "v_nom" not in df_nodes.columns:
            df_nodes["v_nom"] = params.default_v_nom
        network.import_components_from_dataframe(df_nodes.set_index("name"), "Bus")

        df_lines = read_excel(
            input_dir(f"lines;source={params.nodes}.xlsx"), sheet_var="year"
        )
        if "geometry" in df_lines.columns:
            df_lines = df_lines.drop(columns="geometry")
        df_lines = select_subset(df_lines, var="year", vals=params.years)
        df_lines = adjust_lifetime_to_periods(df_lines, years=params.years)
        for col in ["name", "bus0", "bus1"]:
            df_lines[col] = "PL_" + df_lines[col].astype(str)
        network.import_components_from_dataframe(df_lines.set_index("name"), "Line")
        logging.info("Added domestic nodes and lines")

        # Foreign nodes and links
        if params.imports or params.exports:
            df_neighbors = read_excel(input_dir("neighbors.xlsx"))
            if "v_nom" not in df_neighbors.columns:
                df_neighbors["v_nom"] = params.default_v_nom
            network.import_components_from_dataframe(
                df_neighbors.set_index("name"), "Bus"
            )

            df_links = read_excel(
                input_dir(f"interconnectors;source={params.interconnectors}.xlsx")
            )
            # Determine build_year and lifetime parameters
            df_links["build_year"] = df_links["build_year"].fillna(
                params.default_build_year
            )
            df_links["retire_year"] = df_links["retire_year"].fillna(np.inf)
            df_links["lifetime"] = df_links["retire_year"] - df_links["build_year"]
            if params.decommission_year_inclusive:
                df_links["lifetime"] += 1

            # Keep only components that are active in the considered period
            year_min, year_max = min(params.years), max(params.years)
            df_links = df_links[
                (df_links["build_year"] <= year_min)
                & (year_max < df_links["build_year"] + df_links["lifetime"])
            ]

            # Remove links with zero capacity
            df_links = df_links[df_links["p_nom"] > 0]

            # Scale links
            df_links["p_nom"] *= params.trade_factor

            # Define relevant columns
            df_links = df_links[
                ["name", "bus0", "bus1", "carrier", "build_year", "p_nom"]
            ]

            # Split into export and import links
            df_exports = df_links[df_links["bus0"].str.startswith("PL")]
            df_imports = df_links[df_links["bus1"].str.startswith("PL")]

            # TODO: add voivodeship buses later such that we do not need to remove full bus information
            df_exports = df_exports.copy()
            df_imports = df_imports.copy()
            df_exports["bus0"] = network.buses.index[0]
            df_imports["bus1"] = network.buses.index[0]

            if params.dynamic_trade_prices:
                # Demand and load
                df_demand = read_excel(
                    input_dir(
                        f"neighbors_demand;source={params.neighbors_capacity_demand}.xlsx"
                    )
                )
                df_demand = df_demand[["Country", *params.years]]
                df_demand = (
                    df_demand.set_index("Country").transpose().reset_index(names="year")
                )

                df_load = read_excel(
                    input_dir(
                        f"neighbors_load_profile;source={params.neighbors_load_profile}.xlsx"
                    ),
                    sheet_var="year",
                )
                df_load = select_subset(
                    df_load, var="year", vals=[params.load_profile_year]
                ).drop(columns="year")

                df_load = calculate_load_timeseries(
                    df_load,
                    df_demand,
                    network,
                    temporal_resolution=params.temporal_resolution,
                )

                # Attach load to foreign nodes
                df_load = df_load.set_index(["period", "timestep"])
                foreign_nodes = df_load.columns
                df_load = df_load.rename(columns=lambda s: s + " load")
                network.madd("Load", df_load.columns, bus=foreign_nodes, p_set=df_load)

                # Generators
                df_capacity = read_excel(
                    data_dir(
                        "input",
                        f"neighbors_capacities;source={params.neighbors_capacity_demand}.xlsx",
                    ),
                    sheet_var="Country",
                )
                df_capacity = df_capacity.drop(columns="Unit")  # GW
                df_capacity = df_capacity.melt(
                    id_vars=["Country", "Technology"],
                    var_name="year",
                    value_name="p_nom",
                )
                df_capacity["p_nom"] = df_capacity["p_nom"] * 1000  # GW -> MW
                df_capacity["type"] = "capacity"
                df_capacity = df_capacity.rename(columns={"Technology": "category"})

                df_capacity = process_aggregate_capacity_data(
                    df_capacity,
                    area_column="Country",
                    source_technology=params.technology_data,
                    source_hydro_utilization=params.hydro_utilization,
                )

                # Spatial attribution
                df_capacity["bus"] = df_capacity["Country"]
                # Unique name
                df_capacity["name"] = (
                    df_capacity["Country"]
                    + " "
                    + df_capacity["category"]
                    + " "
                    + df_capacity["build_year"].astype(str)
                )
                # Keep only capacities that are active in the considered period
                year_min, year_max = params.years[0], params.years[-1]
                df_capacity = df_capacity[
                    (df_capacity["build_year"] <= year_min)
                    & (year_max < df_capacity["build_year"] + df_capacity["lifetime"])
                ]

                dfs_capacity_factors = process_utilization_profiles(
                    source_renewable=params.renewable_utilization_profiles,
                    source_chp=params.chp_utilization_profiles,
                    network=network,
                    weather_year=params.load_profile_year,
                    temporal_resolution=params.temporal_resolution,
                    domestic=False,
                )

                df_srmc = process_srmc_data(
                    df_capacity,
                    years=params.years,
                    source_prices=params.prices,
                    source_technology=params.technology_data,
                    srmc_dsr=params.srmc_dsr,
                    srmc_wind=params.srmc_wind,
                    srmc_pv=params.srmc_pv,
                    srmc_only_JWCD=params.srmc_only_JWCD,
                    random_seed=params.random_seed + 1,
                )

                df_srmc.to_csv(srmc_file, index=False)

                add_generators(network, df_capacity, dfs_capacity_factors, df_srmc)
                add_storage(network, df_capacity, dfs_capacity_factors, df_srmc)

            else:
                file = data_dir(
                    "input", f"trade_prices;source={params.trade_prices}.xlsx"
                )
                df_tp = read_excel(file)
                df_tp = (
                    df_tp.set_index("Price Type").transpose().reset_index(names="year")
                )

                nodes = network.buses.index
                foreign_nodes = nodes[~nodes.str.startswith("PL")]

                # Results do not depend on the magnitude of the generation capacity or load as long as they are large
                network.madd(
                    "Generator",
                    foreign_nodes + " generator",
                    bus=foreign_nodes,
                    p_nom=200000,
                )

                network.madd(
                    "Load",
                    foreign_nodes + " load",
                    bus=foreign_nodes,
                    p_set=100000,
                )

                # TODO: marginal cost should be time-dependent
                df_imports = pd.merge(
                    df_imports,
                    df_tp.rename(
                        columns={"Electricity Import Price [PLN/MWh]": "marginal_cost"}
                    )[["year", "marginal_cost"]],
                    left_on="build_year",
                    right_on="year",
                    how="left",
                ).drop(columns="year")

                df_exports = pd.merge(
                    df_exports,
                    df_tp.rename(
                        columns={"Electricity Export Price [PLN/MWh]": "marginal_cost"}
                    )[["year", "marginal_cost"]],
                    left_on="build_year",
                    right_on="year",
                    how="left",
                ).drop(columns="year")
                df_exports["marginal_cost"] *= -1

            if params.imports:
                network.import_components_from_dataframe(
                    df_imports.set_index("name"), "Link"
                )
            if params.exports:
                network.import_components_from_dataframe(
                    df_exports.set_index("name"), "Link"
                )

            if params.dynamic_trade_prices:
                # At the moment no scenario for the energy system in UA
                # TODO: remove the UA node in the input data instead
                network.buses = network.buses.drop(index="UA")
                network.links = network.links[~network.links.index.str.contains("UA")]

            logging.info("Added foreign nodes, links, generators, and loads")

        # Generators and storage units

        df_units = process_utility_units_data(
            source_combustion=params.combustion_units,
            source_renewable=params.renewable_units,
            source_storage=params.storage_units,
            source_technology=params.technology_data,
            decommission_year_inclusive=params.decommission_year_inclusive,
            warm_reserve_sources=params.warm_reserve_sources,
            cold_reserve_sources=params.cold_reserve_sources,
        )
        # Spatial attribution
        df_units["bus"] = network.buses.index[0]  # TODO: attribute to the closest node
        # Keep only units that are active in the considered period
        year_min, year_max = params.years[0], params.years[-1]
        df_units = df_units[
            (df_units["build_year"] <= year_min)
            & (year_max < df_units["build_year"] + df_units["lifetime"])
        ]
        # Unique name up to category
        # df_units["name"] = df_units["name"] + " " + df_units["carrier"]

        # Load scenario for capacity investments by technology
        df_capacity = read_excel(
            data_dir(
                "input",
                f"capacity_investments;source={params.capacity_investments}.xlsx",
            ),
            sheet_var="category",
        )

        df_capacity = df_capacity[df_capacity["Voivodeship"] != "ALL"]
        to_drop = [
            col
            for col in ["Unit", "References", "Comments"]
            if col in df_capacity.columns
        ]
        df_capacity = df_capacity.drop(columns=to_drop)
        df_capacity = df_capacity.melt(
            id_vars=["Voivodeship", "category", "type"],
            var_name="year",
            value_name="p_nom",
        ).fillna(0)

        df_capacity = process_aggregate_capacity_data(
            df_capacity,
            area_column="Voivodeship",
            source_technology=params.technology_data,
            source_hydro_utilization=params.hydro_utilization,
            discount_rate=params.discount_rate,
            extendable_technologies=params.extendable_technologies,
            active_investment_years=params.years,
            extend_from_zero=params.extend_from_zero,
            warm_reserve_sources=params.warm_reserve_sources,
            cold_reserve_sources=params.cold_reserve_sources,
            enforce_bio=params.enforce_bio,
            industrial_utilization=params.industrial_utilization,
        )
        # Spatial attribution
        df_capacity["bus"] = network.buses.index[0]  # TODO: distribute among buses
        # Unique name
        df_capacity["name"] = (
            df_capacity["Voivodeship"]
            + " "
            + df_capacity["category"]
            + " "
            + df_capacity["build_year"].astype(str)
        )  # + "_" + df_capacity["bus"]
        # Keep only capacities that are active in the considered period
        df_capacity = df_capacity[
            (df_capacity["build_year"] <= year_min)
            & (year_max < df_capacity["build_year"] + df_capacity["lifetime"])
        ]

        df_generators_storage = pd.concat([df_units, df_capacity])
        df_generators_storage["Country"] = "PL"

        dfs_capacity_factors = process_utilization_profiles(
            source_renewable=params.renewable_utilization_profiles,
            source_chp=params.chp_utilization_profiles,
            network=network,
            weather_year=params.load_profile_year,
            temporal_resolution=params.temporal_resolution,
            correction_factor_wind_old=params.correction_factor_wind_old,
            correction_factor_wind_new=params.correction_factor_wind_new,
        )

        df_srmc = process_srmc_data(
            df_generators_storage,
            years=params.years,
            source_prices=params.prices,
            source_technology=params.technology_data,
            srmc_dsr=params.srmc_dsr,
            srmc_wind=params.srmc_wind,
            srmc_pv=params.srmc_pv,
            srmc_only_JWCD=params.srmc_only_JWCD,
            random_seed=params.random_seed,
        )

        # Save SRMC data
        if srmc_file.exists():
            df_srmc_combined = pd.concat([df_srmc, pd.read_csv(srmc_file)], axis=1)
            df_srmc_combined.to_csv(srmc_file, index=False)
        else:
            df_srmc.to_csv(srmc_file, index=False)

        add_generators(network, df_generators_storage, dfs_capacity_factors, df_srmc)
        add_storage(network, df_generators_storage, dfs_capacity_factors, df_srmc)
        logging.info("Added generators and storage units")

        # Spatial aggregation
        if params.grid_resolution == "copper_plate":
            nodes = network.buses.index
            country = nodes.str[:2]
            busmap = pd.Series(country, index=nodes, name="Cluster")

            with create_fictional_line(network, busmap):
                clustering = get_clustering_from_busmap(network, busmap)
                clustering.network.remove("Line", "1")
                clustering.network.remove("Bus", "fictional")

            # TODO: aggregate links?

            # Inherit the same investment periods
            clustering.network.investment_periods = network.investment_periods
            clustering.network.investment_period_weightings = (
                network.investment_period_weightings
            )

            network = clustering.network

            linemap = clustering.linemap
            linemap = linemap.drop(index="fictional")
            busmap = clustering.busmap

            logging.info("Aggregated the network")

        # Add co2 emissions per carrier
        df_carrier = (
            df_generators_storage.rename(
                columns={"Fuel CO2 emission factor [tCO2/MWh_t]": "co2_emissions"}
            )[["carrier", "co2_emissions"]]
            .groupby("carrier")
            .mean()
            .round(3)
            .fillna(0)
        ).reset_index()
        # Add max growth per carrier
        df_max_growth = read_excel(
            input_dir(f"capacity_max_growth;source={params.capacity_max_growth}.xlsx")
        )
        df_max_growth["carrier"] = df_max_growth["technology"].str[:-5]
        df_carrier = pd.merge(df_carrier, df_max_growth, on="carrier", how="left")
        df_carrier["max_growth"] = (
            df_carrier["max_growth"] * params.extension_years
        ).fillna(inf)

        df_carrier = df_carrier.set_index("carrier")
        attributes = [
            "co2_emissions",
            "max_growth",
        ]
        kwargs = {
            key: df_carrier[key] for key in attributes if key in df_carrier.columns
        }
        network.madd("Carrier", df_carrier.index, **kwargs)

        # Capacity constraints
        df_cp = read_excel(
            input_dir(f"capacity_potentials;source={params.capacity_potentials}.xlsx"),
            sheet_var="carrier",
        )
        df_cp = df_cp[df_cp["Voivodeship"] != "ALL"]

        df_cp = df_cp[["Voivodeship", "carrier", *params.years]].melt(
            id_vars=["Voivodeship", "carrier"], var_name="year", value_name="value"
        )
        df_cp["column"] = (
            "nom_max_" + df_cp["carrier"] + "_" + df_cp["year"].astype(str)
        )
        df_cp = df_cp.pivot(
            index="Voivodeship", columns="column", values="value"
        ).round(3)

        network.voivodeships = df_cp
        network.voivodeships.index.name = "voivodeship"

        # Assign voivodeships to capacities
        # TODO: set it in the input data
        for components in [network.generators, network.storage_units]:
            components["voivodeship"] = components.index.str.split(" ", n=1).str[0]
            components.loc[
                (components["carrier"] == "Wind offshore")
                & components["bus"].str.startswith("PL"),
                "voivodeship",
            ] = "offshore"
            components.loc[
                ~components["voivodeship"].isin(network.voivodeships.index),
                "voivodeship",
            ] = np.nan

        # Demand and load
        df_demand = read_excel(input_dir(f"demand;source={params.demand}.xlsx"))
        df_demand = df_demand[["Demand Type", *params.years]]
        df_demand = (
            df_demand.set_index("Demand Type").transpose().reset_index(names="year")
        )
        df_demand = df_demand[["year", "Electricity [TWh]"]].rename(
            columns={"Electricity [TWh]": "PL"}
        )
        df_demand["PL"] *= params.demand_correction

        df_load = read_excel(
            input_dir(f"load_profile;source={params.load_profile}.xlsx"),
            sheet_var="year",
        )
        df_load = select_subset(df_load, var="year", vals=[params.load_profile_year])
        df_load = df_load.drop(columns="year")

        df_load = calculate_load_timeseries(
            df_load, df_demand, network, temporal_resolution=params.temporal_resolution
        )

        # TODO: Distribute load
        nodes = network.buses.index
        nodes = nodes[nodes == "PL"]
        df_load = df_load.assign(
            **{f"{node} load": df_load["PL"] / len(nodes) for node in nodes}
        )

        # Attach load to nodes
        df_load = df_load.set_index(["period", "timestep"])[nodes + " load"]
        network.madd("Load", df_load.columns, bus=nodes, p_set=df_load)

        # Add virtual DSR (optional)
        if params.virtual_dsr:
            add_virtual_dsr(network, srmc_dsr=params.srmc_dsr)

        logging.info("Added the load")

        network.name = params.scenario

        # Save inputs
        network.export_to_csv_folder(runs_dir(params.scenario, "input"))
        network.voivodeships.to_csv(
            runs_dir(params.scenario, "input", "voivodeships.csv")
        )
        busmap.to_csv(runs_dir(params.scenario, "input", "busmap.csv"))
        linemap.to_csv(runs_dir(params.scenario, "input", "linemap.csv"))

    else:
        # Load inputs
        network = pypsa.Network()
        network.import_from_csv_folder(runs_dir(params.scenario, "input"))
        network.voivodeships = pd.read_csv(
            runs_dir(params.scenario, "input", "voivodeships.csv"), index_col=0
        )

    if dry:
        return
    # Do the computation

    def extra_functionality(network, snapshots):
        p_set_constraint(network, snapshots)
        maximum_annual_capacity_factor(network, snapshots)
        minimum_annual_capacity_factor(network, snapshots)
        warm_reserve_flag = False
        if (
            params.warm_reserve_need_per_demand
            + params.warm_reserve_need_per_pv
            + params.warm_reserve_need_per_wind
            > 0
        ):
            warm_reserve_flag = True
            warm_reserve(
                network,
                snapshots,
                warm_reserve_need_per_demand=params.warm_reserve_need_per_demand,
                warm_reserve_need_per_pv=params.warm_reserve_need_per_pv,
                warm_reserve_need_per_wind=params.warm_reserve_need_per_wind,
                max_r_over_p=params.max_r_over_p,
                hours_per_timestep=int(
                    params.temporal_resolution[:-1]
                ),  # e.g. 1H -> 1,
            )
        if params.cold_reserve_need_per_import > 0:
            cold_reserve(
                network,
                snapshots,
                cold_reserve_need_per_demand=params.cold_reserve_need_per_demand,
                cold_reserve_need_per_import=params.cold_reserve_need_per_import,
                warm_reserve=warm_reserve_flag,
            )
        maximum_capacity_per_voivodeship(network, snapshots)
        maximum_growth_per_carrier(network, snapshots)

    if params.mode == "lopf":
        network.lopf(
            multi_investment_periods=len(params.years) > 1,
            solver_name=params.solver,
            pyomo=False,
            extra_functionality=extra_functionality,
        )

    # Save outputs
    network.export_to_csv_folder(runs_dir(params.scenario, "output"))
