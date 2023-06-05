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
from pypsa_pl.helper_functions import (
    filter_lifetimes,
    select_subset,
    repeat_over_periods,
    update_lifetime,
)
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
    maximum_capacity_per_area,
    maximum_growth_per_carrier,
    maximum_snsp,
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
        self.buses = "voivodeships"
        self.lines = "OSM+PSE_existing"
        self.line_factor = 1
        self.line_types = "pypsa_pl_v2.1"
        self.links = None
        self.neighbors = "pypsa_pl_v2.1"
        self.interconnectors = "pypsa_pl_v2.1"
        self.default_build_year = 2020
        self.demand = "PSE_2022"
        self.demand_correction = 1.0
        self.srmc_only_JWCD = False
        self.load_profile = "entsoe"
        self.load_profile_year = 2012
        self.neighbors_load_profile = "PECD3"
        self.neighbors_capacity_demand = "TYNDP_2022"
        self.sectors = ["electricity"]
        self.thermal_units = "energy.instrat.pl"
        self.renewable_units = "pypsa_pl_v2.1"
        self.storage_units = "pypsa_pl_v2.1"
        self.aggregate_units = "baseline"
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
        self.use_pyomo = False
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
        self.warm_reserve_categories = [
            "JWCD",
            "Hydro PSH",
            "Battery large",
            "Battery large 1h",
            "Battery large 4h",
        ]
        self.cold_reserve_categories = ["JWCD"]
        self.warm_reserve_need_per_demand = 0.09
        self.warm_reserve_need_per_pv = 0.15
        self.warm_reserve_need_per_wind = 0.10
        self.cold_reserve_need_per_demand = 0.09
        self.cold_reserve_need_per_import = 1.0
        self.max_r_over_p = 1.0
        self.max_snsp = 0.75
        self.ns_carriers = [
            "Wind onshore",
            "Wind offshore",
            "PV roof",
            "PV ground",
            "DC",
            "Battery large",
            "Battery small",
        ]
        self.unit_commitment_categories = None
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

    # Assume df_demand has areas as columns
    df_demand = df_demand.melt(
        id_vars="year", var_name="area", value_name="annual_demand"
    )
    df_demand["country"] = df_demand["area"].str[:2]

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
        df_load.pivot(index=["period", "timestep"], columns="area", values="p_set")
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
        category="Virtual DSR",
        carrier="Virtual DSR",
        marginal_cost=srmc_dsr,
    )


def run_pypsa_pl(params=Params(), use_cache=False, dry=False):
    logging.info(f"Running PyPSA-PL for parameters: {params}")

    if not use_cache:
        os.makedirs(runs_dir(params.scenario, "input"), exist_ok=True)

        network = make_network(
            temporal_resolution=params.temporal_resolution,
            years=params.years,
            discount_rate=params.discount_rate,
        )
        logging.info("Created the network")

        # Domestic nodes
        df_buses = read_excel(input_dir(f"buses;source={params.buses}.xlsx"))
        network.import_components_from_dataframe(df_buses.set_index("name"), "Bus")
        network.add("Bus", "PL")

        # Line types
        df_line_types = read_excel(
            input_dir(f"line_types;source={params.line_types}.xlsx")
        )
        network.import_components_from_dataframe(
            df_line_types.set_index("name"), "LineType"
        )

        # Domestic lines
        if params.lines:
            df_lines = read_excel(input_dir(f"lines;source={params.lines}.xlsx"))
            df_lines["s_max_pu"] = params.line_factor
            network.import_components_from_dataframe(df_lines.set_index("name"), "Line")

        # Domestic links
        if params.links:
            df_intercon = read_excel(input_dir(f"links;source={params.links}.xlsx"))
            network.import_components_from_dataframe(
                df_intercon.set_index("name"), "Link"
            )

        logging.info("Added domestic nodes and lines")

        # Foreign nodes and links
        if params.imports or params.exports:
            df_neighbors = read_excel(
                input_dir(f"neighbors;source={params.neighbors}.xlsx")
            )
            network.import_components_from_dataframe(
                df_neighbors.set_index("name"), "Bus"
            )

            df_intercon = read_excel(
                input_dir(f"interconnectors;source={params.interconnectors}.xlsx")
            )
            # Determine build_year and lifetime parameters
            df_intercon = update_lifetime(
                df_intercon,
                default_build_year=params.default_build_year,
                decommission_year_inclusive=params.decommission_year_inclusive,
            )

            # Keep only components that are active in the considered period
            df_intercon = filter_lifetimes(df_intercon, years=params.years)

            # Remove links with zero capacity
            df_intercon = df_intercon[df_intercon["p_nom"] > 0]

            # Scale links
            df_intercon["p_nom"] *= params.trade_factor

            # Split into export and import links
            df_exports = df_intercon[
                df_intercon["bus0"].str.startswith("PL")
                & df_intercon["bus1"].isin(df_neighbors["name"])
            ].copy()
            df_imports = df_intercon[
                df_intercon["bus1"].str.startswith("PL")
                & df_intercon["bus0"].isin(df_neighbors["name"])
            ].copy()

            if params.dynamic_trade_prices:
                # Demand and load
                df_demand = read_excel(
                    input_dir(
                        f"neighbors_demand;source={params.neighbors_capacity_demand}.xlsx"
                    )
                )
                df_demand = df_demand[["area", *params.years]]
                df_demand = (
                    df_demand.set_index("area").transpose().reset_index(names="year")
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

                # Attach load to neighbor buses
                df_load = df_load.set_index(["period", "timestep"])
                neighbor_buses = df_load.columns
                df_load = df_load.rename(columns=lambda s: s + " electricity demand")
                network.madd(
                    "Load",
                    df_load.columns,
                    bus=neighbor_buses,
                    carrier="AC",
                    p_set=df_load,
                )

                # Generators
                df_capacity = read_excel(
                    data_dir(
                        "input",
                        f"neighbors_capacities;source={params.neighbors_capacity_demand}.xlsx",
                    ),
                    sheet_var="area",
                )
                df_capacity = df_capacity.melt(
                    id_vars=["area", "category", "carrier", "technology"],
                    var_name="build_year",
                    value_name="p_nom",
                )
                df_capacity["p_nom"] = df_capacity["p_nom"] * 1000  # GW -> MW
                df_capacity["value_type"] = "total"

                df_capacity = process_aggregate_capacity_data(
                    df_capacity,
                    source_technology=params.technology_data,
                    source_hydro_utilization=params.hydro_utilization,
                    decommission_year_inclusive=params.decommission_year_inclusive,
                    default_build_year=params.default_build_year,
                )

                # Spatial attribution
                df_capacity["bus"] = df_capacity["area"]
                # Unique name
                df_capacity["name"] = (
                    df_capacity["area"]
                    + " "
                    + df_capacity["technology"]
                    + " "
                    + df_capacity["build_year"].astype(str)
                )
                # Keep only capacities that are active in the considered period
                df_capacity = filter_lifetimes(df_capacity, years=params.years)

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

                add_generators(network, df_capacity, dfs_capacity_factors, df_srmc)
                add_storage(network, df_capacity, dfs_capacity_factors, df_srmc)

            else:
                file = data_dir(
                    "input", f"trade_prices;source={params.trade_prices}.xlsx"
                )
                df_tp = read_excel(file)
                df_tp = (
                    df_tp.set_index("Price Type")
                    .transpose()
                    .reset_index(names="year")
                    .set_index("year")
                )

                buses = network.buses.index
                neighbor_buses = buses[~buses.str.startswith("PL")]

                # Results do not depend on the magnitude of the generation capacity or load as long as they are large
                network.madd(
                    "Generator",
                    neighbor_buses + " generator",
                    bus=neighbor_buses,
                    carrier="AC",
                    p_nom=200000,
                )

                network.madd(
                    "Load",
                    neighbor_buses + " electricity demand",
                    bus=neighbor_buses,
                    carrier="AC",
                    p_set=100000,
                )

                # TODO: marginal cost should be time-dependent
                assert len(params.years) == 1
                year = params.years[0]
                df_imports["marginal_cost"] = df_tp.loc[
                    year, "Electricity Import Price [PLN/MWh]"
                ]
                df_exports["marginal_cost"] = -df_tp.loc[
                    year, "Electricity Export Price [PLN/MWh]"
                ]

            if params.imports:
                network.import_components_from_dataframe(
                    df_imports.set_index("name"), "Link"
                )
            if params.exports:
                network.import_components_from_dataframe(
                    df_exports.set_index("name"), "Link"
                )

            logging.info("Added foreign nodes, links, generators, and loads")

        # Generators and storage units
        df_units = process_utility_units_data(
            source_thermal=params.thermal_units,
            source_renewable=params.renewable_units,
            source_storage=params.storage_units,
            source_technology=params.technology_data,
            default_build_year=params.default_build_year,
            decommission_year_inclusive=params.decommission_year_inclusive,
            warm_reserve_categories=params.warm_reserve_categories,
            cold_reserve_categories=params.cold_reserve_categories,
            unit_commitment_categories=params.unit_commitment_categories,
            hours_per_timestep=int(params.temporal_resolution[:-1]),  # e.g. 1H -> 1,
        )

        # Spatial attribution
        df_units["bus"] = df_units["area"]
        # Unique name
        df_units["name"] = df_units["name"] + " " + df_units["technology"]

        # Keep only capacities that are active in the considered period
        df_units = filter_lifetimes(df_units, years=params.years)

        # Aggregate units
        df_capacity = read_excel(
            data_dir(
                "input",
                f"aggregate_units;source={params.aggregate_units}.xlsx",
            ),
            sheet_var="group",
        )

        id_vars = [
            "group",
            "area",
            "category",
            "carrier",
            "technology",
        ]
        df_capacity = df_capacity.melt(
            id_vars=id_vars + ["value_type"],
            var_name="build_year",
            value_name="p_nom",
        ).fillna(0)
        df_capacity["build_year"] = df_capacity["build_year"].astype(int)

        # Aggregate additions if possible
        df_addition, df_total = (
            df_capacity[df_capacity["value_type"] == "addition"].copy(),
            df_capacity[df_capacity["value_type"] == "total"].copy(),
        )
        build_years = df_addition["build_year"].unique()
        periodic_years = {
            int(year)
            for year in params.extension_years
            * np.ceil(build_years / params.extension_years)
        }
        incommensurate_years = {
            year for year in params.years if year % params.extension_years != 0
        }
        aggregation_years = list(sorted(periodic_years.union(incommensurate_years)))
        df_addition["build_year_agg"] = aggregation_years[-1]
        for year in aggregation_years[-2::-1]:
            df_addition.loc[df_addition["build_year"] <= year, "build_year_agg"] = year
        df_addition = (
            df_addition.groupby(by=id_vars + ["build_year_agg"])
            .agg({"p_nom": "sum"})
            .reset_index()
        )
        df_addition = df_addition.rename(columns={"build_year_agg": "build_year"})
        df_capacity = pd.concat([df_addition, df_total])

        df_capacity = process_aggregate_capacity_data(
            df_capacity,
            source_technology=params.technology_data,
            source_hydro_utilization=params.hydro_utilization,
            discount_rate=params.discount_rate,
            extendable_technologies=params.extendable_technologies,
            active_investment_years=params.years,
            extend_from_zero=params.extend_from_zero,
            warm_reserve_categories=params.warm_reserve_categories,
            cold_reserve_categories=params.cold_reserve_categories,
            enforce_bio=params.enforce_bio,
            industrial_utilization=params.industrial_utilization,
            decommission_year_inclusive=params.decommission_year_inclusive,
            default_build_year=params.default_build_year,
        )
        # Spatial attribution
        df_capacity["bus"] = df_capacity["area"]
        # Unique name
        df_capacity["name"] = (
            df_capacity["area"]
            + " "
            + df_capacity["group"]
            + " "
            + df_capacity["build_year"].astype(str)
        )
        # Keep only capacities that are active in the considered period
        df_capacity = filter_lifetimes(df_capacity, years=params.years)

        df_generators_storage = pd.concat([df_units, df_capacity])

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

        add_generators(network, df_generators_storage, dfs_capacity_factors, df_srmc)
        add_storage(network, df_generators_storage, dfs_capacity_factors, df_srmc)
        logging.info("Added generators and storage units")

        # Demand and load
        df_demands = read_excel(
            input_dir(f"exogenous_demand;source={params.demand}.xlsx"),
            sheet_var="sector",
        )
        df_demands = df_demands[["sector", "area", *[str(y) for y in params.years]]]

        for sector in params.sectors:
            if sector != "electricity":
                logging.warning(
                    f"Sectors other than electricity are not implemented yet."
                )
                continue
            df_demand = select_subset(df_demands, var="sector", vals=[sector]).drop(
                columns="sector"
            )

            df_demand = (
                df_demand.set_index("area").transpose().reset_index(names="year")
            )
            df_demand["year"] = df_demand["year"].astype(int)
            df_demand[
                [col for col in df_demand.columns if col.startswith("PL")]
            ] *= params.demand_correction

            df_load = read_excel(
                input_dir(f"load_profile;source={params.load_profile}.xlsx"),
                sheet_var="year",
            )
            df_load = select_subset(
                df_load, var="year", vals=[params.load_profile_year]
            )
            df_load = df_load.drop(columns="year")

            df_load = calculate_load_timeseries(
                df_load,
                df_demand,
                network,
                temporal_resolution=params.temporal_resolution,
            )

            # Attach load to buses
            df_load = df_load.set_index(["period", "timestep"])
            buses = df_load.columns
            df_load = df_load.rename(columns=lambda s: s + " electricity demand")
            network.madd(
                "Load", df_load.columns, bus=buses, carrier="AC", p_set=df_load
            )

        logging.info("Added the load")

        # Spatial aggregation
        assert params.grid_resolution in ["copper_plate", "voivodeships"]
        if params.grid_resolution == "copper_plate":
            buses = network.buses.index
            country = buses.str[:2]
            busmap = pd.Series(country, index=buses, name="Cluster")

            with create_fictional_line(network, busmap):
                clustering = get_clustering_from_busmap(network, busmap)
                clustering.network.remove("Line", "1")
                clustering.network.remove("Bus", "fictional")

            # Inherit the same investment periods
            clustering.network.investment_periods = network.investment_periods
            clustering.network.investment_period_weightings = (
                network.investment_period_weightings
            )

            network = clustering.network

            network.linemap = clustering.linemap.drop(index="fictional")
            network.busmap = clustering.busmap

            logging.info("Aggregated the network")
        else:
            network.remove("Bus", "PL")

        # Add co2 emissions per carrier
        df_carrier = (
            df_generators_storage.rename(
                columns={"fuel_co2_emissions": "co2_emissions"}
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
        df_max_growth["carrier"] = df_max_growth["technology"]
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

        df_cp = df_cp[["area", "carrier", *[str(y) for y in params.years]]].melt(
            id_vars=["area", "carrier"], var_name="year", value_name="value"
        )
        df_cp["column"] = (
            "nom_max_" + df_cp["carrier"] + "_" + df_cp["year"].astype(str)
        )
        df_cp = df_cp.pivot(index="area", columns="column", values="value").round(3)

        network.areas = df_cp
        network.areas.index.name = "area"

        # Add virtual DSR (optional)
        if params.virtual_dsr:
            add_virtual_dsr(network, srmc_dsr=params.srmc_dsr)

        network.meta = vars(params)
        network.name = params.scenario

        # Save inputs
        network.export_to_csv_folder(runs_dir(params.scenario, "input"))
        network.areas.to_csv(runs_dir(params.scenario, "input", "areas.csv"))
        if hasattr(network, "busmap") and hasattr(network, "linemap"):
            network.busmap.to_csv(runs_dir(params.scenario, "input", "busmap.csv"))
            network.linemap.to_csv(runs_dir(params.scenario, "input", "linemap.csv"))

    else:
        # Load inputs
        network = pypsa.Network()
        network.import_from_csv_folder(runs_dir(params.scenario, "input"))
        network.areas = pd.read_csv(
            runs_dir(params.scenario, "input", "areas.csv"), index_col=0
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
        maximum_capacity_per_area(network, snapshots)
        maximum_growth_per_carrier(network, snapshots)
        if params.max_snsp < 1:
            maximum_snsp(
                network,
                snapshots,
                max_snsp=params.max_snsp,
                ns_carriers=params.ns_carriers,
            )

    solver_options = {}
    if params.solver == "gurobi":
        solver_options = {
            "Threads": 4,
            "Method": 2,  # barrier
            "PreSolve": -1 if params.unit_commitment_categories is None else 1,
            "PrePasses": -1 if params.unit_commitment_categories is None else 1,
            "MIPFocus": 1, # affects only MIPs
            "MIPGap": 0.99, # affects only MIPs - basically return the first feasible solution
            "Crossover": 0,
            "BarConvTol": 1e-6,
            "FeasibilityTol": 1e-5,
            "AggFill": 0,
            "PreDual": 0,
            "GURO_PAR_BARDENSETHRESH": 200,
            "Seed": 0,
        }
    if params.solver == "highs":
        solver_options = {
            "threads": 4,
            "solver": "ipm",
            "run_crossover": "off",
            "small_matrix_value": 1e-6,
            "large_matrix_value": 1e9,
            "primal_feasibility_tolerance": 1e-5,
            "dual_feasibility_tolerance": 1e-5,
            "ipm_optimality_tolerance": 1e-4,
            "parallel": "on",
            "random_seed": 0,
        }

    if params.mode == "lopf":
        network.lopf(
            multi_investment_periods=len(params.years) > 1,
            solver_name=params.solver,
            solver_options=solver_options,
            pyomo=params.use_pyomo,
            extra_functionality=extra_functionality,
        )

    # Save outputs
    network.export_to_csv_folder(runs_dir(params.scenario, "output"))

    df_stat = network.statistics(
        groupby=pypsa.statistics.get_bus_and_carrier
    ).reset_index()
    df_stat.columns = ["Component"] + [col for col in df_stat.columns[1:]]
    df_stat.to_csv(runs_dir(params.scenario, "output", "statistics.csv"), index=False)
