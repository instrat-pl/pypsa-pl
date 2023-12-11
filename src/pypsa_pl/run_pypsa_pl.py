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
from pypsa_pl.helper_functions import calculate_statistics
from pypsa_pl.make_network import make_network, make_custom_component_attrs
from pypsa_pl.process_capacity_factors_data import process_utilization_profiles
from pypsa_pl.process_generator_storage_data import (
    process_utility_units_data,
    process_aggregate_capacity_data,
    process_sectoral_capacity_data,
    assign_reserve_assumptions,
)
from pypsa_pl.process_srmc_data import process_srmc_data
from pypsa_pl.add_generators_and_storage import (
    add_generators,
    add_storage_units,
    add_links,
    add_stores,
)
from pypsa_pl.custom_constraints import (
    maximum_annual_capacity_factor,
    minimum_annual_capacity_factor,
    make_warm_reserve_constraints,
    cold_reserve,
    maximum_capacity_per_area,
    maximum_snsp,
    technology_bundle_constraints,
    bev_ext_constraint,
    bev_charge_constraint,
    chp_dispatch_constraint,
    chp_ext_constraint,
    maximum_resistive_heater_small_production,
    store_dispatch_constraint,
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
        self.years = [2025]
        self.imports = True
        self.exports = True
        self.trade_factor = 0.7
        self.dynamic_trade_prices = True
        self.trade_prices = "pypsa_pl_v1"
        self.fixed_trade_flows = None
        self.buses = "voivodeships"
        self.lines = "OSM+PSE_existing"
        self.line_factor = 1
        self.line_types = "pypsa_pl_v2.1"
        self.links = None
        self.neighbors = "pypsa_pl_v2.1"
        self.interconnectors = "pypsa_pl_v2.1"
        self.default_build_year = 2020
        self.demand = "instrat_ambitious_copper_plate"
        self.electricity_demand_correction = 1.0
        self.srmc_only_JWCD = False
        self.electricity_load_profile = "entsoe"
        self.heat_load_profile = "degree_days+bdew"
        self.light_vehicles_load_profile = "gddkia"
        self.load_profile_year = 2012
        self.neighbors_load_profile = "PECD3"
        self.neighbors_capacity_demand = "TYNDP_2022"
        self.sectors = ["electricity", "heat", "light vehicles", "hydrogen"]
        self.thermal_units = None  # "energy.instrat.pl"
        self.renewable_units = None # "pypsa_pl_v2.1"
        self.storage_units = "pypsa_pl_v2.1"
        self.aggregate_units = "instrat_ambitious_copper_plate"
        self.sectoral_units = "instrat_ambitious_copper_plate"
        self.capacity_potentials = "instrat_ambitious"
        self.capacity_max_growth = "instrat_ambitious"
        self.fuel_potentials = "instrat_2021"
        self.technology_data = "instrat_2023"
        self.hydro_utilization = "entsoe_2020"
        self.renewable_utilization_profiles = "PECD3+EMHIRES"
        self.chp_utilization_profiles = "regression"
        self.heat_pump_efficiency_profiles = "instrat"
        self.prices = "instrat_2023"
        self.scenario = "instrat_ambitious"
        self.solver = "highs"
        self.mode = "lopf"
        self.repeat_with_fixed_capacities = True
        self.decommission_year_inclusive = True
        self.srmc_wind = 8.0
        self.srmc_pv = 1.0
        self.srmc_dsr = 1200
        self.enforce_bio = 0
        self.enforce_jwcd = 0
        self.industrial_utilization = 0.5
        self.correction_factor_wind_old = 0.91
        self.correction_factor_wind_new = 1.09
        self.bev_availability_max = 0.95
        self.bev_availability_mean = 0.8
        self.v2g_factor = 0.25
        self.minimum_bev_charge_level = 0.75
        self.minimum_bev_charge_hour = 6
        self.srmc_v2g = 50
        self.biogas_substrate_price_factor = 1
        self.oil_to_petroleum_product_price_ratio = 2
        self.electricity_distribution_loss = 0.05
        self.district_heating_loss = 0.12
        self.district_heating_distribution_cost = 72
        self.share_district_heating_min = 0.32
        self.share_district_heating_max = 0.32
        self.share_biomass_boiler_min = 0.2
        self.share_biomass_boiler_max = 0.2
        self.max_resistive_to_heat_pump_ratio = 0.02
        self.discount_rate = 0.03
        self.exclude_technologies = None
        self.extendable_technologies = None
        self.decommission_only_technologies = None
        self.extend_from_zero = False
        self.primary_reserve_categories = [
            "JWCD",
            "Battery large",
            "Battery large 1h",
            "Battery large 2h",
            "Battery large 4h",
        ]
        self.primary_reserve_minutes = 0.5
        self.tertiary_reserve_categories = [
            "JWCD",
            "Hydro PSH",
            "Battery large",
            "Battery large 1h",
            "Battery large 2h",
            "Battery large 4h",
        ]
        self.tertiary_reserve_minutes = 30
        self.cold_reserve_categories = ["JWCD"]
        self.warm_reserve_need_per_demand = 0
        self.warm_reserve_need_per_pv = 0
        self.warm_reserve_need_per_wind = 0
        self.primary_reserve_factor = 0
        self.tertiary_reserve_factor = 0
        self.cold_reserve_need_per_demand = 0
        self.cold_reserve_need_per_import = 0
        self.max_r_over_p = 1.0
        self.max_snsp = 1.0
        self.ns_carriers = [
            "Wind onshore",
            "Wind offshore",
            "PV roof",
            "PV ground",
            "DC",
            "Battery large",
            "Battery small",
            "BEV V2G",
        ]
        self.unit_commitment_categories = None
        self.linearized_unit_commitment = True
        self.thermal_constraints_factor = 1
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


def calculate_load_timeseries(
    df_load, df_demand, network, temporal_resolution, merge_on="country"
):
    # Aggregate the load profile to the temporal resolution of the snapshots
    df_load = (
        df_load.groupby(pd.Grouper(key="hour", freq=temporal_resolution))
        .mean()
        .reset_index()
    )
    df_load = df_load.rename(columns={"hour": "timestep"})

    # Merge snapshots and load profile on month, day, and hour
    df_load = repeat_over_periods(df_load, network)

    # Merge snapshots and demand on year
    df_load = df_load.melt(
        id_vars=["period", "timestep"], var_name=merge_on, value_name="load_profile"
    )

    # Assume df_demand has areas as columns
    df_demand = df_demand.melt(
        id_vars="year", var_name="area", value_name="annual_demand"
    )
    df_demand["country"] = df_demand["area"].str[:2]

    df_load = pd.merge(
        df_load,
        df_demand,
        left_on=["period", merge_on],
        right_on=["year", merge_on],
        how="right",
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
            network.loads[["bus", "p_set"]],
            how="outer",
            left_index=True,
            right_index=True,
        )
        .groupby("bus")
        .sum()
        .max(axis=1)
    )
    buses = network.buses.loc[max_load_per_bus.index]
    category = ("Virtual DSR " + buses["carrier"]).str.replace(" AC", "")
    network.madd(
        "Generator",
        buses.index,
        suffix=f" Virtual DSR",
        bus=buses.index,
        p_nom=max_load_per_bus,
        category=category,
        carrier=category,
        technology=category,
        marginal_cost=100 * srmc_dsr,
    )


def run_pypsa_pl(params=Params(), use_cache=False, dry=False):
    logging.info(f"Running PyPSA-PL for parameters: {params}")

    if not use_cache:
        os.makedirs(runs_dir(params.scenario, "input"), exist_ok=True)

        reserves = []
        if params.primary_reserve_categories:
            reserves += ["primary_up", "primary_down"]
        if params.tertiary_reserve_categories:
            reserves += ["tertiary_up"]
        custom_component_attrs = make_custom_component_attrs(reserves=reserves)

        network = make_network(
            temporal_resolution=params.temporal_resolution,
            years=params.years,
            discount_rate=params.discount_rate,
            custom_component_attrs=custom_component_attrs,
        )
        logging.info("Created the network")

        # Domestic nodes
        df_buses = read_excel(input_dir(f"buses;source={params.buses}.xlsx"))
        network.import_components_from_dataframe(df_buses.set_index("name"), "Bus")
        network.add(
            "Bus",
            "PL",
            carrier="AC",
            v_nom=400,
            x=df_buses["x"].mean(),
            y=df_buses["y"].mean(),
        )
        df_buses = network.buses.copy()
        df_buses["v_nom"] = np.nan
        # TODO: treat biogas as a separate sector
        for sector in params.sectors + ["biogas"]:
            if sector == "electricity":
                continue
            buses = [sector]
            if sector == "light vehicles":
                buses += ["BEV"]
            if sector == "heat":
                buses += ["District heating", "Heat pump small"]
            for bus in buses:
                carrier = sector
                df_buses_sector = df_buses.copy()
                df_buses_sector.index += f" {bus}"
                df_buses_sector["carrier"] = bus
                network.import_components_from_dataframe(df_buses_sector, "Bus")

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
            df_links = read_excel(input_dir(f"links;source={params.links}.xlsx"))
            df_links["p0_sign"] = 1
            network.import_components_from_dataframe(df_links.set_index("name"), "Link")

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

            # Set sign to 1
            df_intercon["p0_sign"] = 1

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

                df_load = pd.read_csv(
                    input_dir(
                        "timeseries",
                        f"neighbors_load_profile;source={params.neighbors_load_profile};year={params.load_profile_year}.csv",
                    ),
                    parse_dates=["hour"],
                )
                # Normalize the profile such that it sums to 1
                df_load = df_load.set_index("hour")
                df_load = df_load / df_load.sum(axis=0)
                df_load = df_load.reset_index()

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
                    value_name="nom",
                )
                df_capacity["nom"] = df_capacity["nom"] * 1000  # GW -> MW
                df_capacity["value_type"] = "total"
                df_capacity["build_year"] = df_capacity["build_year"].astype(int)

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
                    source_bev=params.light_vehicles_load_profile,
                    source_heat_pump=params.heat_pump_efficiency_profiles,
                    network=network,
                    weather_year=params.load_profile_year,
                    temporal_resolution=params.temporal_resolution,
                    domestic=False,
                )

                df_capacity, df_srmc = process_srmc_data(
                    df_capacity,
                    years=params.years,
                    source_prices=params.prices,
                    source_technology=params.technology_data,
                    srmc_dsr=params.srmc_dsr,
                    srmc_wind=params.srmc_wind,
                    srmc_pv=params.srmc_pv,
                    srmc_only_JWCD=params.srmc_only_JWCD,
                    random_seed=params.random_seed + 1,
                    oil_to_petroleum_product_price_ratio=params.oil_to_petroleum_product_price_ratio,
                    biogas_substrate_price_factor=params.biogas_substrate_price_factor,
                )

                add_generators(network, df_capacity, dfs_capacity_factors, df_srmc)
                add_storage_units(network, df_capacity, dfs_capacity_factors, df_srmc)

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

            if params.fixed_trade_flows is not None:
                file = data_dir(
                    "input",
                    "timeseries",
                    f"interconnector_utilization;source={params.fixed_trade_flows}.csv",
                )
                df_trade_flows = pd.read_csv(file, parse_dates=["timestep"])
                df_trade_flows = df_trade_flows.set_index(["period", "timestep"])
                network.import_series_from_dataframe(df_trade_flows, "Link", "p_set")

            logging.info("Added foreign nodes, links, generators, and loads")

        # Generators and storage units

        if params.extendable_technologies is None:
            params.extendable_technologies = []
        if params.decommission_only_technologies is None:
            params.decommission_only_technologies = []

        df_units = process_utility_units_data(
            source_thermal=params.thermal_units,
            source_renewable=params.renewable_units,
            source_storage=params.storage_units,
            source_technology=params.technology_data,
            default_build_year=params.default_build_year,
            decommission_year_inclusive=params.decommission_year_inclusive,
            decommission_only_technologies=[*params.decommission_only_technologies],
            unit_commitment_categories=params.unit_commitment_categories,
            linearized_unit_commitment=params.linearized_unit_commitment,
            thermal_constraints_factor=params.thermal_constraints_factor,
            enforce_jwcd=params.enforce_jwcd,
            hours_per_timestep=int(params.temporal_resolution[:-1]),  # e.g. 1H -> 1,
            sectors=params.sectors,
        )

        if params.grid_resolution == "copper_plate":
            df_units["area"] = "PL"

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
        df_capacity["sector"] = "electricity"
        df_capacity["technology_bundle"] = "electricity"

        # Sectoral units
        df_capacity_sectors = read_excel(
            data_dir(
                "input",
                f"sectoral_units;source={params.sectoral_units}.xlsx",
            ),
            sheet_var="group",
        )
        df_capacity = pd.concat([df_capacity, df_capacity_sectors])

        # TODO: do not allow for missing technology_bundle
        df_capacity["technology_bundle"] = df_capacity["technology_bundle"].fillna(
            "electricity"
        )

        id_vars = [
            "group",
            "area",
            "sector",
            "category",
            "carrier",
            "technology",
            "technology_bundle",
        ]
        df_capacity = df_capacity.melt(
            id_vars=id_vars + ["value_type"],
            var_name="build_year",
            value_name="nom",
        )
        df_capacity["nom"] = df_capacity["nom"].fillna(0)
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
            df_addition.groupby(by=id_vars + ["value_type", "build_year_agg"])
            .agg({"nom": "sum"})
            .reset_index()
        )
        df_addition = df_addition.rename(columns={"build_year_agg": "build_year"})
        df_capacity = pd.concat([df_addition, df_total])

        is_sectoral = (df_capacity["sector"] != "electricity") | (
            df_capacity["technology"].isin(["Biogas plant", "Biogas storage"])
        )

        df_capacity_electric = process_aggregate_capacity_data(
            df_capacity[~is_sectoral],
            source_technology=params.technology_data,
            source_hydro_utilization=params.hydro_utilization,
            discount_rate=params.discount_rate,
            extendable_technologies=[*params.extendable_technologies],
            decommission_only_technologies=[*params.decommission_only_technologies],
            active_investment_years=params.years,
            extend_from_zero=params.extend_from_zero,
            enforce_bio=params.enforce_bio,
            enforce_jwcd=params.enforce_jwcd,
            industrial_utilization=params.industrial_utilization,
            decommission_year_inclusive=params.decommission_year_inclusive,
            default_build_year=params.default_build_year,
            sectors=params.sectors,
        )

        if len(params.sectors) > 1:
            df_capacity_sectoral = process_sectoral_capacity_data(
                df_capacity[is_sectoral],
                source_technology=params.technology_data,
                discount_rate=params.discount_rate,
                extendable_technologies=[*params.extendable_technologies],
                decommission_only_technologies=[*params.decommission_only_technologies],
                active_investment_years=params.years,
                extend_from_zero=params.extend_from_zero,
                decommission_year_inclusive=params.decommission_year_inclusive,
                default_build_year=params.default_build_year,
                v2g_factor=params.v2g_factor,
            )

            df_capacity_sectoral = df_capacity_sectoral[
                df_capacity_sectoral["sector"].isin(params.sectors)
            ]

            df_capacity = pd.concat([df_capacity_electric, df_capacity_sectoral])
        else:
            df_capacity = df_capacity_electric

        # Bus attribution
        df_capacity["bus"] = df_capacity["area"]

        # Biogas
        is_biogas_sector = df_capacity["technology"].isin(
            ["Biogas plant", "Biogas storage"]
        )
        df_capacity.loc[is_biogas_sector, "bus"] = (
            df_capacity.loc[is_biogas_sector, "area"] + " biogas"
        )

        biogas_to_electricity = df_capacity["technology"].isin(
            ["Biogas engine", "Biogas engine CHP"]
        )
        df_capacity.loc[biogas_to_electricity, "bus0"] = (
            df_capacity.loc[biogas_to_electricity, "area"] + " biogas"
        )
        df_capacity.loc[biogas_to_electricity, "bus1"] = df_capacity.loc[
            biogas_to_electricity, "area"
        ]

        # Other sectors

        is_sectoral = df_capacity["sector"] != "electricity"
        df_capacity.loc[is_sectoral, "bus"] = (
            df_capacity.loc[is_sectoral, "area"]
            + " "
            + df_capacity.loc[is_sectoral, "sector"]
        )

        bev_battery = df_capacity["technology"].isin(["BEV battery"])
        df_capacity.loc[bev_battery, "bus"] = (
            df_capacity.loc[bev_battery, "area"] + " BEV"
        )

        electricity_to_hydrogen = df_capacity["technology"].isin(["Electrolyser"])

        df_capacity.loc[electricity_to_hydrogen, "bus0"] = df_capacity.loc[
            electricity_to_hydrogen, "area"
        ]
        df_capacity.loc[electricity_to_hydrogen, "bus1"] = (
            df_capacity.loc[electricity_to_hydrogen, "area"] + " hydrogen"
        )

        hydrogen_to_electricity = df_capacity["technology"].isin(
            ["Hydrogen CCGT", "Hydrogen OCGT", "Hydrogen CCGT CHP", "Hydrogen OCGT CHP"]
        )
        df_capacity.loc[hydrogen_to_electricity, "bus0"] = (
            df_capacity.loc[hydrogen_to_electricity, "area"] + " hydrogen"
        )
        df_capacity.loc[hydrogen_to_electricity, "bus1"] = df_capacity.loc[
            hydrogen_to_electricity, "area"
        ]

        electricity_to_bev = df_capacity["technology"].isin(["BEV charger"])
        df_capacity.loc[electricity_to_bev, "bus0"] = df_capacity.loc[
            electricity_to_bev, "area"
        ]
        df_capacity.loc[electricity_to_bev, "bus1"] = (
            df_capacity.loc[electricity_to_bev, "area"] + " BEV"
        )

        bev_to_electricity = df_capacity["technology"].isin(["BEV V2G"])
        df_capacity.loc[bev_to_electricity, "bus0"] = (
            df_capacity.loc[bev_to_electricity, "area"] + " BEV"
        )
        df_capacity.loc[bev_to_electricity, "bus1"] = df_capacity.loc[
            bev_to_electricity, "area"
        ]

        bev_to_light_vehicles = df_capacity["technology"].isin(["BEV"])
        df_capacity.loc[bev_to_light_vehicles, "bus0"] = (
            df_capacity.loc[bev_to_light_vehicles, "area"] + " BEV"
        )
        df_capacity.loc[bev_to_light_vehicles, "bus1"] = (
            df_capacity.loc[bev_to_light_vehicles, "area"] + " light vehicles"
        )

        district_heat_storage = df_capacity["technology"].isin(["Heat storage large"])
        df_capacity.loc[district_heat_storage, "bus"] = (
            df_capacity.loc[district_heat_storage, "area"] + " District heating"
        )

        district_heat_generators = df_capacity["technology"].isin(
            ["Conventional heating plant"]
        ) | df_capacity["technology"].str.contains("CHP heat output")
        df_capacity.loc[district_heat_generators, "bus"] = (
            df_capacity.loc[district_heat_generators, "area"] + " District heating"
        )

        ### df_units
        # TODO: merge df_units and df_capacity earlier
        district_heat_generators = df_units["technology"].str.contains(
            "CHP heat output"
        )
        df_units.loc[district_heat_generators, "bus"] = (
            df_units.loc[district_heat_generators, "area"] + " District heating"
        )
        ###

        electricity_to_district_heat = df_capacity["technology"].isin(
            ["Heat pump large", "Resistive heater large"]
        )
        df_capacity.loc[electricity_to_district_heat, "bus0"] = df_capacity.loc[
            electricity_to_district_heat, "area"
        ]
        df_capacity.loc[electricity_to_district_heat, "bus1"] = (
            df_capacity.loc[electricity_to_district_heat, "area"] + " District heating"
        )

        heat_pump_small_storage = df_capacity["technology"].isin(["Heat storage small"])
        df_capacity.loc[heat_pump_small_storage, "bus"] = (
            df_capacity.loc[heat_pump_small_storage, "area"] + " Heat pump small"
        )

        electricity_to_heat_pump_small = df_capacity["technology"].isin(
            ["Heat pump small", "Resistive heater small"]
        )
        df_capacity.loc[electricity_to_heat_pump_small, "bus0"] = df_capacity.loc[
            electricity_to_heat_pump_small, "area"
        ]
        df_capacity.loc[electricity_to_heat_pump_small, "bus1"] = (
            df_capacity.loc[electricity_to_heat_pump_small, "area"] + " Heat pump small"
        )

        # TODO: define distribution and transmission nodes and link between them with losses instead
        df_capacity.loc[
            electricity_to_hydrogen
            | electricity_to_bev
            | electricity_to_district_heat
            | electricity_to_heat_pump_small,
            "efficiency",
        ] *= (
            1 - params.electricity_distribution_loss
        )

        # Those links should be defined by the output and not the input capacity. Hence:
        # (1) swap bus0 with bus1
        # (2) set p_min_pu to -p_max_pu and p_max_pu to 0
        # (3) set p_min_pu_annual to -p_max_pu_annual and p_max_pu_annual to nan
        # (4) set efficiency to 1/efficiency
        # (4) remember that p variable of the link should always be preceded by minus sign

        reverse_links = (
            biogas_to_electricity
            | electricity_to_hydrogen
            | hydrogen_to_electricity
            | bev_to_light_vehicles
            | electricity_to_district_heat
            | electricity_to_heat_pump_small
        )
        df_capacity.loc[reverse_links, ["bus0", "bus1"]] = df_capacity.loc[
            reverse_links, ["bus1", "bus0"]
        ].values
        df_capacity.loc[reverse_links, "p_min_pu"] = -df_capacity.loc[
            reverse_links, "p_max_pu"
        ]
        df_capacity.loc[reverse_links, "p_max_pu"] = 0
        # Set p_min_pu_annual to -p_max_pu_annual and p_max_pu_annual to -p_min_pu_annual
        for attr in ["p_min_pu_annual", "p_max_pu_annual"]:
            if attr not in df_capacity.columns:
                df_capacity[attr] = np.nan
        df_capacity.loc[
            reverse_links, ["p_min_pu_annual", "p_max_pu_annual"]
        ] = -df_capacity.loc[
            reverse_links, ["p_max_pu_annual", "p_min_pu_annual"]
        ].values
        df_capacity.loc[reverse_links, "p_min_pu_annual"] = df_capacity.loc[
            reverse_links, "p_min_pu_annual"
        ].fillna(-1)
        df_capacity.loc[reverse_links, "p_max_pu_annual"] = df_capacity.loc[
            reverse_links, "p_max_pu_annual"
        ].fillna(0)

        df_capacity.loc[reverse_links, "efficiency"] = (
            1 / df_capacity.loc[reverse_links, "efficiency"]
        ).round(3)
        df_capacity.loc[reverse_links, "p0_sign"] = -1

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

        df_generators_storage = assign_reserve_assumptions(
            df_generators_storage,
            primary_reserve_categories=params.primary_reserve_categories,
            tertiary_reserve_categories=params.tertiary_reserve_categories,
            primary_reserve_minutes=params.primary_reserve_minutes,
            tertiary_reserve_minutes=params.tertiary_reserve_minutes,
            cold_reserve_categories=params.cold_reserve_categories,
        )

        dfs_capacity_factors = process_utilization_profiles(
            source_renewable=params.renewable_utilization_profiles,
            source_chp=params.chp_utilization_profiles,
            source_bev=params.light_vehicles_load_profile,
            source_heat_pump=params.heat_pump_efficiency_profiles,
            network=network,
            weather_year=params.load_profile_year,
            temporal_resolution=params.temporal_resolution,
            correction_factor_wind_old=params.correction_factor_wind_old,
            correction_factor_wind_new=params.correction_factor_wind_new,
            bev_availability_max=params.bev_availability_max,
            bev_availability_mean=params.bev_availability_mean,
            electricity_distribution_loss=params.electricity_distribution_loss,
        )

        df_generators_storage, df_srmc = process_srmc_data(
            df_generators_storage,
            years=params.years,
            source_prices=params.prices,
            source_technology=params.technology_data,
            srmc_dsr=params.srmc_dsr,
            srmc_wind=params.srmc_wind,
            srmc_pv=params.srmc_pv,
            srmc_v2g=params.srmc_v2g,
            srmc_only_JWCD=params.srmc_only_JWCD,
            random_seed=params.random_seed,
            oil_to_petroleum_product_price_ratio=params.oil_to_petroleum_product_price_ratio,
            biogas_substrate_price_factor=params.biogas_substrate_price_factor,
        )

        if params.exclude_technologies is not None:
            df_generators_storage = df_generators_storage[
                ~df_generators_storage["technology"].isin(params.exclude_technologies)
            ]

        add_generators(
            network,
            df_generators_storage,
            dfs_capacity_factors,
            df_srmc,
            fix_chp="heat" not in params.sectors,
        )
        add_storage_units(network, df_generators_storage, dfs_capacity_factors, df_srmc)
        add_stores(network, df_generators_storage, dfs_capacity_factors, df_srmc)
        add_links(network, df_generators_storage, dfs_capacity_factors, df_srmc)

        logging.info("Added generators and storage units")

        # Demand and load
        df_demands = read_excel(
            input_dir(f"exogenous_demand;source={params.demand}.xlsx"),
            sheet_var="sector",
        )
        df_demands = df_demands[["sector", "area", *[str(y) for y in params.years]]]

        for sector in params.sectors:
            subsectors = [sector]
            if sector == "heat":
                subsectors = ["heat - space", "heat - water"]

            loads = []
            for subsector in subsectors:
                df_demand = select_subset(
                    df_demands, var="sector", vals=[subsector]
                ).drop(columns="sector")
                df_demand = (
                    df_demand.set_index("area").transpose().reset_index(names="year")
                )
                df_demand["year"] = df_demand["year"].astype(int)
                if sector == "electricity":
                    df_demand[
                        [col for col in df_demand.columns if col.startswith("PL")]
                    ] *= params.electricity_demand_correction

                # Hydrogen and water heating loads are assumed constant
                if subsector not in ["hydrogen", "heat - water"]:
                    subsector_prefix = subsector.replace(" - ", "_").replace(" ", "_")
                    source_load_profile = getattr(
                        params, f"{sector.replace(' ', '_')}_load_profile"
                    )

                    df_load = pd.read_csv(
                        input_dir(
                            "timeseries",
                            f"{subsector_prefix}_load_profile;source={source_load_profile};year={params.load_profile_year}.csv",
                        ),
                        parse_dates=["hour"],
                    )
                    # Normalize the profile such that it sums to 1
                    df_load = df_load.set_index("hour")
                    df_load = df_load / df_load.sum(axis=0)
                    df_load = df_load.reset_index()

                    # Only heat load profile is spatially heterogeneous
                    df_load = calculate_load_timeseries(
                        df_load,
                        df_demand,
                        network,
                        temporal_resolution=params.temporal_resolution,
                        merge_on="country" if sector != "heat" else "area",
                    )

                    df_load = df_load.set_index(["period", "timestep"])
                    loads.append(df_load)
                else:
                    if len(params.years) == 1:
                        load = df_demand.set_index("year").loc[
                            params.years[0], :
                        ] / len(network.snapshots)
                        hours_per_timestep = int(params.temporal_resolution[:-1])
                        load = load / hours_per_timestep * 1e6
                        loads.append(load)
                    else:
                        assert False, "Multi-year simulations are not supported."

            df_load = sum(loads)
            # If there is no time dependence, load is a series
            if isinstance(df_load, pd.Series):
                df_load = df_load.to_frame().transpose()

            # Attach load to buses
            if sector == "electricity":
                buses = df_load.columns
                carrier = "AC"
            else:
                buses = [f"{bus} {sector}" for bus in df_load.columns]
                carrier = sector
            df_load = df_load.rename(columns=lambda s: s + f" {sector} demand")
            network.madd(
                "Load",
                df_load.columns,
                bus=buses,
                carrier=carrier,
                p_set=df_load if len(df_load) > 1 else df_load.iloc[0],
            )

        logging.info("Added loads")

        # Spatial aggregation
        assert params.grid_resolution in ["copper_plate", "voivodeships"]
        if params.grid_resolution == "copper_plate":
            buses = network.buses.index.to_series()
            carrier = network.buses.carrier
            country_sector = buses.str[:2]
            is_sector = carrier != "AC"
            country_sector[is_sector] = (
                country_sector[is_sector] + " " + carrier[is_sector]
            )

            busmap = pd.Series(country_sector, index=buses, name="Cluster")

            with create_fictional_line(network, busmap):
                clustering = get_clustering_from_busmap(network, busmap)
                clustering.network.remove("Line", "1")
                clustering.network.remove("Bus", "fictional")

            # Inherit the same investment periods
            clustering.network.investment_periods = network.investment_periods
            clustering.network.investment_period_weightings = (
                network.investment_period_weightings
            )

            # Preserve custom components
            for c in ["Generator", "StorageUnit"]:
                for reserve in reserves:
                    clustering.network.pnl(c)[f"r_{reserve}"] = network.pnl(c)[
                        f"r_{reserve}"
                    ]

            # Preserve p_set for links if present
            if "p_set" in network.links_t.keys():
                clustering.network.links_t["p_set"] = network.links_t["p_set"]

            network = clustering.network

            network.linemap = clustering.linemap.drop(index="fictional")
            network.busmap = clustering.busmap

            logging.info("Aggregated the network")
        else:
            buses_with_loads = network.loads.bus.unique()
            buses_without_loads = network.buses.index.difference(buses_with_loads)
            network.mremove("Bus", buses_without_loads)

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

        # Add biogas_substrate and biomass_straw attributes
        df_carrier.loc[
            df_carrier.index.intersection(["Biogas", "Biogas plant"]),
            "biogas_substrate",
        ] = 1
        df_carrier.loc[
            df_carrier.index.intersection(["Biomass straw"]), "biomass_straw"
        ] = 1

        # Treat foreign biogas and biomass as seperate carriers
        network.generators.loc[
            ~network.generators.area.str.startswith("PL")
            & network.generators.carrier.isin(["Biogas", "Biomass straw"]),
            "carrier",
        ] += " foreign"

        attributes = [
            "co2_emissions",
            "biomass_straw",
            "biogas_substrate",
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

        areas_carriers = df_generators_storage[["area", "carrier"]].drop_duplicates()
        df_cp = pd.merge(df_cp, areas_carriers, on=["area", "carrier"], how="inner")

        df_cp["column"] = (
            "nom_max_" + df_cp["carrier"] + "_" + df_cp["year"].astype(str)
        )

        df_cp = df_cp.pivot(index="area", columns="column", values="value").round(3)

        network.areas = df_cp
        network.areas.index.name = "area"

        # Fuel and CO2 potentials
        df_fp = read_excel(
            input_dir(f"fuel_potentials;source={params.fuel_potentials}.xlsx"),
            sheet_var="carrier_attribute",
        )
        df_fp = df_fp[
            ["area", "carrier_attribute", *[str(y) for y in params.years]]
        ].melt(
            id_vars=["area", "carrier_attribute"], var_name="year", value_name="value"
        )
        df_fp["value"] *= 1e6  # TWh -> MWh, MtCO2 -> tCO2
        # TODO: allow constraints per area
        df_fp = df_fp[df_fp["area"] == "PL"].drop(columns="area")
        df_fp.index = df_fp["carrier_attribute"] + "_" + df_fp["year"].astype(str)
        network.madd(
            "GlobalConstraint",
            df_fp.index,
            type="primary_energy",
            carrier_attribute=df_fp["carrier_attribute"],
            investment_period=df_fp["year"].astype(int),
            sense="<=",
            constant=df_fp["value"],
        )

        # Technology bundles
        df_tb = df_generators_storage[["technology_bundle", "sector"]].drop_duplicates()
        # Consider technology bundle functionality only for heat and light vehicle sectors
        technology_bundles_dict = {
            sector: df_tb.loc[df_tb["sector"] == sector, "technology_bundle"].tolist()
            for sector in ["heat", "light vehicles"]
        }

        # Add extra links from district heat and heat pump small to heat
        if "heat" in params.sectors:
            heat_buses = network.buses[network.buses.carrier == "heat"].index
            max_demand = network.loads_t.p_set[heat_buses + " demand"].max()
            for bundle in ["District heating", "Heat pump small"]:
                efficiency = 1
                marginal_cost = 0
                if bundle == "District heating":
                    efficiency -= params.district_heating_loss
                    marginal_cost = params.district_heating_distribution_cost
                network.madd(
                    "Link",
                    heat_buses.str.replace(" heat", f" {bundle} output"),
                    bus0=heat_buses.str.replace(" heat", f" {bundle}"),
                    bus1=heat_buses,
                    p_min_pu=0,
                    p_max_pu=1,
                    p_nom=max_demand.rename(
                        index=lambda x: x.replace(" heat demand", f" {bundle} output")
                    )
                    / efficiency,
                    efficiency=efficiency,
                    marginal_cost=marginal_cost,
                    carrier=f"{bundle} output",
                    technology_bundle=bundle,
                    p0_sign=1,
                )

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

    # Do the computation

    def extra_functionality(network, snapshots):
        maximum_annual_capacity_factor(network, snapshots)
        minimum_annual_capacity_factor(network, snapshots)
        store_dispatch_constraint(network, snapshots)
        # WARNING: warm reserves are not supported now
        warm_reserve_names = []
        # if (
        #     params.warm_reserve_need_per_demand
        #     + params.warm_reserve_need_per_pv
        #     + params.warm_reserve_need_per_wind
        #     > 0
        # ) and (params.primary_reserve_categories or params.tertiary_reserve_categories):
        #     # Only one kind of warm reserves can be active at a time
        #     # TODO: allow more than one kind of warm reserves to be active at a time
        #     assert len(warm_reserve_names) < 1
        #     if params.primary_reserve_categories:
        #         directions = ["up", "down"]
        #         reserve_factor = params.primary_reserve_factor
        #         warm_reserve_names += ["primary"]
        #     if params.tertiary_reserve_categories:
        #         directions = ["up"]
        #         reserve_factor = params.tertiary_reserve_factor
        #         warm_reserve_names += ["tertiary"]
        #     make_warm_reserve_constraints(
        #         reserve_name=warm_reserve_names[0],
        #         reserve_factor=reserve_factor,
        #         directions=directions,
        #     )(
        #         network,
        #         snapshots,
        #         warm_reserve_need_per_demand=params.warm_reserve_need_per_demand,
        #         warm_reserve_need_per_pv=params.warm_reserve_need_per_pv,
        #         warm_reserve_need_per_wind=params.warm_reserve_need_per_wind,
        #         max_r_over_p=params.max_r_over_p,
        #         hours_per_timestep=int(
        #             params.temporal_resolution[:-1]
        #         ),  # e.g. 1H -> 1,
        #     )
        if (
            (params.cold_reserve_need_per_import > 0)
            or (params.cold_reserve_need_per_demand > 0)
        ) and params.cold_reserve_categories:
            cold_reserve(
                network,
                snapshots,
                cold_reserve_need_per_demand=params.cold_reserve_need_per_demand,
                cold_reserve_need_per_import=params.cold_reserve_need_per_import,
                warm_reserve_names=warm_reserve_names,
            )
        maximum_capacity_per_area(network, snapshots)
        if params.max_snsp < 1:
            maximum_snsp(
                network,
                snapshots,
                max_snsp=params.max_snsp,
                ns_carriers=params.ns_carriers,
            )
        if "light vehicles" in params.sectors:
            bev_ext_constraint(network, snapshots)
            bev_charge_constraint(
                network,
                snapshots,
                hour=params.minimum_bev_charge_hour,
                charge_level=params.minimum_bev_charge_level,
            )
        if "heat" in params.sectors:
            chp_dispatch_constraint(network, snapshots)
            chp_ext_constraint(network, snapshots)
            maximum_resistive_heater_small_production(
                network, snapshots, params.max_resistive_to_heat_pump_ratio
            )
        if len(params.sectors) > 1:
            technology_bundle_constraints(
                network,
                snapshots,
                technology_bundles_dict,
                district_heating_range=(
                    params.share_district_heating_min,
                    params.share_district_heating_max,
                ),
                biomass_boiler_range=(
                    params.share_biomass_boiler_min,
                    params.share_biomass_boiler_max,
                ),
            )

        if params.grid_resolution == "copper_plate":
            # Remove all subnetworks as done here: https://github.com/PyPSA/PyPSA/blob/0555a5b4cc8c26995f2814927cf250e928825cba/pypsa/components.py#LL1279C1-L1283C20
            # Otherwise this code causes trouble: https://github.com/PyPSA/PyPSA/blob/0555a5b4cc8c26995f2814927cf250e928825cba/pypsa/optimization/optimize.py#L431
            for sub_network in network.sub_networks.index:
                obj = network.sub_networks.at[sub_network, "obj"]
                network.remove("SubNetwork", sub_network)
                del obj
            if "obj" in network.sub_networks.columns:
                network.sub_networks = network.sub_networks.drop(columns="obj")

        constraints = repr(network.model.constraints)
        with open(runs_dir(params.scenario, "input", "constraints.txt"), "w") as f:
            f.write(constraints)

        variables = repr(network.model.variables)
        with open(runs_dir(params.scenario, "input", "variables.txt"), "w") as f:
            f.write(variables)

        if dry:
            exit()

    solver_options = {}
    if params.solver == "gurobi":
        solver_options = {
            "Threads": 4,
            "Method": 2,  # barrier
            "Crossover": 0,
            "BarConvTol": 1e-8,
            "FeasibilityTol": 1e-7,
            "AggFill": 0,
            "PreDual": 0,
            "GURO_PAR_BARDENSETHRESH": 200,
            "NumericFocus": 1,
            "Seed": 0,
        }
        if (
            params.unit_commitment_categories is not None
            and not params.linearized_unit_commitment
        ):
            solver_options = {
                **solver_options,
                "PreSolve": 1,
                "PrePasses": 1,
                "MIPFocus": 1,
                "MIPGap": 0.5,  # basically return the first feasible solution
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
        optimize_kwargs = dict(
            multi_investment_periods=True,
            solver_name=params.solver,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            linearized_unit_commitment=params.linearized_unit_commitment,
            # transmission_losses=0,
        )
        network.optimize(**optimize_kwargs)
        if (
            params.repeat_with_fixed_capacities
            and len(
                params.extendable_technologies + params.decommission_only_technologies
            )
            > 0
        ):
            logging.info("Optimized investments. Optimizing dispatch only now.")
            network.optimize.fix_optimal_capacities()
            network.optimize(**optimize_kwargs)

    # Save outputs
    network.export_to_csv_folder(runs_dir(params.scenario, "output"))

    df_stat = calculate_statistics(network)
    df_stat.to_csv(runs_dir(params.scenario, "output", "statistics.csv"), index=False)

    return network
