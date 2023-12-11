import pandas as pd
import numpy as np
import xarray as xr
import logging

from pypsa.optimization.compat import (
    get_var,
    linexpr,
    define_constraints,
    define_variables,
)
import linopy

from pypsa.descriptors import get_switchable_as_dense


def get_fixed_demand(network, snapshots):
    """Get fixed electricity demand for each snapshot."""

    sector_load_suffices = (
        " heat demand",
        " hydrogen demand",
        " light vehicles demand",
    )
    loads = network.loads_t["p_set"]
    loads = loads[
        [
            name
            for name in loads.columns
            if name.startswith("PL") and not name.endswith(sector_load_suffices)
        ]
    ]
    demand_t = loads.sum(axis=1)
    demand_t = xr.DataArray(demand_t, dims=["snapshot"])
    return demand_t


def get_variable_demand(network, snapshots, scale_factor=1):
    """Get variable electricity demand for each snapshot."""
    demand_t_list = []

    # Demand from charging storage units
    components = network.storage_units
    store_i = components[components["bus"].str.startswith("PL")].index
    if not store_i.empty:
        store_t = (
            get_var(network, "StorageUnit", "p_store")
            .sel({"StorageUnit": store_i})
            .sum("StorageUnit")
        )
        demand_t_list.append(store_t)

    # Demand from export links
    components = network.links
    export_i = components[
        components["bus0"].str.startswith("PL")
        & ~components["bus1"].str.startswith("PL")
    ].index
    if not export_i.empty:
        export_t = get_var(network, "Link", "p").sel({"Link": export_i}).sum("Link")
        demand_t_list.append(export_t)

    # Demand from inter-sectoral links
    sector_bus_suffices = (
        " heat",
        " hydrogen",
        " light vehicles",
        " BEV",
        " biogas",
        " Heat pump small",
        " District heating",
    )
    components = network.links
    # Electricity withdrawal at bus0
    sector_i = components[
        ~components["bus0"].str.endswith(sector_bus_suffices)
        & components["bus1"].str.endswith(sector_bus_suffices)
        & (components["p0_sign"] > 0)
    ].index
    if not sector_i.empty:
        sector_t = get_var(network, "Link", "p").sel({"Link": sector_i}).sum("Link")
        demand_t_list.append(sector_t)
    # Electricity withdrawal at bus1
    sector_i = components[
        components["bus0"].str.endswith(sector_bus_suffices)
        & ~components["bus1"].str.endswith(sector_bus_suffices)
        & (components["p0_sign"] < 0)
    ].index
    if not sector_i.empty:
        efficiency = get_switchable_as_dense(network, "Link", "efficiency")[sector_i]
        efficiency = xr.DataArray(
            efficiency.values,
            coords={"snapshot": efficiency.index, "Link": efficiency.columns},
        )
        sector_t = (-1) * (
            get_var(network, "Link", "p").sel({"Link": sector_i}) * efficiency
        ).sum("Link")
        demand_t_list.append(sector_t)

    # Aggregate variable demand
    demand_t = scale_factor * sum(demand_t_list)
    return demand_t


def get_variable_supply(network, snapshots, scale_factor=1, carriers=None):
    """Get variable electricity supply for each snapshot."""
    supply_t_list = []

    # Supply from generators
    components = network.generators
    generator_i = components[
        components["bus"].str.startswith("PL")
        & (components["carrier"].isin(carriers) if carriers else True)
    ].index
    if not generator_i.empty:
        generator_t = (
            get_var(network, "Generator", "p")
            .sel({"Generator": generator_i})
            .sum("Generator")
        )
        supply_t_list.append(generator_t)

    # Supply from discharging storage units
    components = network.storage_units
    dispatch_i = components[
        components["bus"].str.startswith("PL")
        & (components["carrier"].isin(carriers) if carriers else True)
    ].index
    if not dispatch_i.empty:
        dispatch_t = (
            get_var(network, "StorageUnit", "p_dispatch")
            .sel({"StorageUnit": dispatch_i})
            .sum("StorageUnit")
        )
        supply_t_list.append(dispatch_t)

    # Supply from import links
    components = network.links
    import_i = components[
        ~components["bus0"].str.startswith("PL")
        & components["bus1"].str.startswith("PL")
        & (components["carrier"].isin(carriers) if carriers else True)
    ].index
    if not import_i.empty:
        import_t = get_var(network, "Link", "p").sel({"Link": import_i}).sum("Link")
        supply_t_list.append(import_t)

    # Supply from inter-sectoral links
    sector_bus_suffices = (
        " heat",
        " hydrogen",
        " light vehicles",
        " BEV",
        " biogas",
        " Heat pump small",
        " District heating",
    )
    components = network.links
    # Electricity injection at bus0
    sector_i = components[
        ~components["bus0"].str.endswith(sector_bus_suffices)
        & components["bus1"].str.endswith(sector_bus_suffices)
        & (components["p0_sign"] < 0)
        & (components["carrier"].isin(carriers) if carriers else True)
    ].index
    if not sector_i.empty:
        sector_t = (-1) * get_var(network, "Link", "p").sel({"Link": sector_i}).sum(
            "Link"
        )
        supply_t_list.append(sector_t)
    # Electricity injection at bus
    sector_i = components[
        components["bus0"].str.endswith(sector_bus_suffices)
        & ~components["bus1"].str.endswith(sector_bus_suffices)
        & (components["p0_sign"] > 0)
        & (components["carrier"].isin(carriers) if carriers else True)
    ].index
    if not sector_i.empty:
        efficiency = get_switchable_as_dense(network, "Link", "efficiency")[sector_i]
        efficiency = xr.DataArray(
            efficiency.values,
            coords={"snapshot": efficiency.index, "Link": efficiency.columns},
        )
        sector_t = (
            get_var(network, "Link", "p").sel({"Link": sector_i}) * efficiency
        ).sum("Link")
        supply_t_list.append(sector_t)

    # Aggregate variable supply
    supply_t = scale_factor * sum(supply_t_list)
    return supply_t


def get_fixed_demand_per_sector(network, snapshots, sector):
    """Get fixed sectoral demand for each snapshot and for each bus."""

    # TODO: this will work only if p_set is time-dependent. Fix it.
    loads = network.loads_t["p_set"]
    loads = loads[
        [name for name in loads.columns if name.endswith(f" {sector} demand")]
    ]
    # Rename loads to the buses they are attached to
    loads = loads.rename(columns=lambda x: x.replace(" demand", ""))
    loads.columns.name = "Bus"

    demand_t = xr.DataArray(
        loads.values,
        coords={"snapshot": loads.index, "Bus": loads.columns},
    )
    return demand_t


def get_variable_supply_per_technology_bundle(
    network, snapshots, technology_bundle, sector
):
    """Get supply per technology bundle for each snapshot and for each bus."""

    supply_t_list = []

    # Generators
    components = network.generators
    generator_i = components[
        (components["technology_bundle"] == technology_bundle)
        & components["bus"].str.endswith(sector)
    ].index
    if not generator_i.empty:
        bus = (
            components.loc[generator_i, "bus"]
            .rename("Bus")
            .rename_axis("Generator")
            .to_xarray()
        )
        generator_t = (
            get_var(network, "Generator", "p")
            .sel({"Generator": generator_i})
            .groupby_sum(bus)
        )
        supply_t_list.append(generator_t)

    # Stores
    components = network.stores
    store_i = components[
        (components["technology_bundle"] == technology_bundle)
        & components["bus"].str.endswith(sector)
    ].index
    if not store_i.empty:
        bus = (
            components.loc[store_i, "bus"]
            .rename("Bus")
            .rename_axis("Store")
            .to_xarray()
        )
        store_t = (
            get_var(network, "Store", "p").sel({"Store": store_i}).groupby_sum(bus)
        )
        supply_t_list.append(store_t)

    # Links
    components = network.links
    # Supply injection at bus0
    reverse_link_i = components[
        (components["p0_sign"] < 0)
        & (components["technology_bundle"] == technology_bundle)
        & components["bus0"].str.endswith(sector)
    ].index
    if not reverse_link_i.empty:
        bus = (
            components.loc[reverse_link_i, "bus0"]
            .rename("Bus")
            .rename_axis("Link")
            .to_xarray()
        )
        reverse_link_t = (-1) * get_var(network, "Link", "p").sel(
            {"Link": reverse_link_i}
        ).groupby_sum(bus)
        supply_t_list.append(reverse_link_t)
    # Supply injection at bus1
    link_i = components[
        (components["p0_sign"] > 0)
        & (components["technology_bundle"] == technology_bundle)
        & components["bus1"].str.endswith(sector)
    ].index
    if not link_i.empty:
        bus = (
            components.loc[link_i, "bus1"].rename("Bus").rename_axis("Link").to_xarray()
        )
        efficiency = get_switchable_as_dense(network, "Link", "efficiency")[link_i]
        efficiency = xr.DataArray(
            efficiency.values,
            coords={"snapshot": efficiency.index, "Link": efficiency.columns},
        )
        link_t = (
            get_var(network, "Link", "p").sel({"Link": link_i}) * efficiency
        ).groupby_sum(bus)
        supply_t_list.append(link_t)

    supply_t = sum(supply_t_list)
    # "_term" should be dimension without coordinates
    # TODO: verify if this fix can be dropped in future PyPSA versions
    supply_t = linopy.LinearExpression(
        supply_t.data.drop("_term", dim=None), network.model
    )

    return supply_t


def technology_bundle_constraints(
    network,
    snapshots,
    technology_bundles_dict,
    district_heating_range=(0, 1),
    biomass_boiler_range=(0, 1),
):
    """For selected sectors we assume that each technology bundle has to deliver a fixed share of the demand at each snapshot, i.e. merit order assumption does not hold."""
    for sector, technology_bundles in technology_bundles_dict.items():
        sector_string = sector.replace(" ", "_")
        demand_t = get_fixed_demand_per_sector(network, snapshots, sector).rename(
            {"Bus": f"Bus-{sector_string}"}
        )
        buses_i = network.buses.index
        buses_i = buses_i[buses_i.str.endswith(f" {sector}")].rename(
            f"Bus-{sector_string}"
        )

        for technology_bundle in technology_bundles:
            var_name = f"share_{technology_bundle.replace(' ', '_')}_in_{sector_string}"

            network.buses[var_name + "_opt"] = np.nan

            share_min, share_max = 0, 1
            if technology_bundle == "District heating":
                share_min, share_max = district_heating_range
            if technology_bundle == "Biomass boiler":
                share_min, share_max = biomass_boiler_range

            define_variables(
                network,
                share_min,
                share_max,
                "Bus",
                var_name,
                axes=[buses_i],
            )

            supply_t = get_variable_supply_per_technology_bundle(
                network, snapshots, technology_bundle, sector
            ).rename({"Bus": f"Bus-{sector_string}"})
            bundle_share = get_var(network, "Bus", var_name)
            demand_part_t = linexpr((demand_t, bundle_share))
            define_constraints(
                network,
                supply_t - demand_part_t,
                "==",
                0,
                f"Bus-{sector_string}",
                f"{var_name}_constraint",
            )


def maximum_annual_capacity_factor_ext(network, snapshots):
    for component, components in [
        ("Generator", network.generators),
        ("Link", network.links),
    ]:
        if "p_max_pu_annual" not in components.columns:
            continue
        ext_i = network.get_extendable_i(component)
        constraint_i = ext_i.intersection(
            components[components["p_max_pu_annual"] < components["p_max_pu"]].index
        )
        if constraint_i.empty:
            continue
        p_t = (
            get_var(network, component, "p")
            .rename({component: f"{component}-ext-max_annual"})
            .sel({f"{component}-ext-max_annual": constraint_i})
        )
        p_max_pu_annual = (
            components.loc[constraint_i, "p_max_pu_annual"]
            .rename_axis(f"{component}-ext-max_annual")
            .to_xarray()
        )
        p_nom = (
            get_var(network, component, "p_nom")
            .rename({f"{component}-ext": f"{component}-ext-max_annual"})
            .sel({f"{component}-ext-max_annual": constraint_i})
        )
        N = len(snapshots)
        p_annual = linexpr((1, p_t)).sum("snapshot")
        minus_p_max_annual = linexpr((-N * p_max_pu_annual, p_nom))
        define_constraints(
            network,
            p_annual + minus_p_max_annual,
            "<=",
            0,
            f"{component}-ext-max_annual",
            "maximum_annual_capacity_factor_ext",
        )


def maximum_annual_capacity_factor_non_ext(network, snapshots):
    for component, components in [
        ("Generator", network.generators),
        ("Link", network.links),
    ]:
        if "p_max_pu_annual" not in components.columns:
            continue
        non_ext_i = network.get_non_extendable_i(component)
        constraint_i = non_ext_i.intersection(
            components[components["p_max_pu_annual"] < components["p_max_pu"]].index
        )
        if constraint_i.empty:
            continue
        p_t = (
            get_var(network, component, "p")
            .rename({component: f"{component}-max_annual"})
            .sel({f"{component}-max_annual": constraint_i})
        )
        p_max_pu_annual = components.loc[constraint_i, "p_max_pu_annual"]
        p_nom = components.loc[constraint_i, "p_nom"]
        N = len(snapshots)
        p_annual = linexpr((1, p_t)).sum("snapshot")
        p_annual_max = (
            (N * p_max_pu_annual * p_nom)
            .rename_axis(f"{component}-max_annual")
            .to_xarray()
        )
        define_constraints(
            network,
            p_annual,
            "<=",
            p_annual_max,
            f"{component}-max_annual",
            "maximum_annual_capacity_factor_non_ext",
        )


def maximum_annual_capacity_factor(network, snapshots):
    maximum_annual_capacity_factor_ext(network, snapshots)
    maximum_annual_capacity_factor_non_ext(network, snapshots)


def minimum_annual_capacity_factor_ext(network, snapshots):
    for component, components in [
        ("Generator", network.generators),
        ("Link", network.links),
    ]:
        if "p_min_pu_annual" not in components.columns:
            continue
        ext_i = network.get_extendable_i(component)
        constraint_i = ext_i.intersection(
            components[components["p_min_pu_annual"] > components["p_min_pu"]].index
        )
        if constraint_i.empty:
            continue
        p_t = (
            get_var(network, component, "p")
            .rename({component: f"{component}-ext-min_annual"})
            .sel({f"{component}-ext-min_annual": constraint_i})
        )
        p_min_pu_annual = (
            components.loc[constraint_i, "p_min_pu_annual"]
            .rename_axis(f"{component}-ext")
            .to_xarray()
        )
        p_nom = (
            get_var(network, component, "p_nom")
            .rename({f"{component}-ext": f"{component}-ext-min_annual"})
            .sel({f"{component}-ext-min_annual": constraint_i})
        )
        N = len(snapshots)
        p_annual = linexpr((1, p_t)).sum("snapshot")
        minus_p_min_annual = linexpr((-N * p_min_pu_annual, p_nom))
        define_constraints(
            network,
            p_annual + minus_p_min_annual,
            ">=",
            0,
            f"{component}-ext-min_annual",
            "minimum_annual_capacity_factor_ext",
        )


def minimum_annual_capacity_factor_non_ext(network, snapshots):
    for component, components in [
        ("Generator", network.generators),
        ("Link", network.links),
    ]:
        if "p_min_pu_annual" not in components.columns:
            continue
        non_ext_i = network.get_non_extendable_i(component)
        constraint_i = non_ext_i.intersection(
            components[components["p_min_pu_annual"] > components["p_min_pu"]].index
        )
        if constraint_i.empty:
            continue
        p_t = (
            get_var(network, component, "p")
            .rename({component: f"{component}-min_annual"})
            .sel({f"{component}-min_annual": constraint_i})
        )
        p_min_pu_annual = components.loc[constraint_i, "p_min_pu_annual"]
        p_nom = components.loc[constraint_i, "p_nom"]
        N = len(snapshots)
        p_annual = linexpr((1, p_t)).sum("snapshot")
        p_annual_min = (
            (N * p_min_pu_annual * p_nom)
            .rename_axis(f"{component}-min_annual")
            .to_xarray()
        )
        define_constraints(
            network,
            p_annual,
            ">=",
            p_annual_min,
            f"{component}-min_annual",
            "minimum_annual_capacity_factor_non_ext",
        )


def minimum_annual_capacity_factor(network, snapshots):
    minimum_annual_capacity_factor_ext(network, snapshots)
    minimum_annual_capacity_factor_non_ext(network, snapshots)


# TODO: allow for different warm reserves at the same time
def make_warm_reserve_non_ext_constraints(reserve_name="warm", direction="up"):
    def warm_reserve_non_ext(
        network,
        snapshots,
        component,
        reserve_units,
        max_r_over_p,
        hours_per_timestep,
    ):
        components = (
            network.generators if component == "Generator" else network.storage_units
        )
        non_ext_i = network.get_non_extendable_i(component)
        constraint_i = non_ext_i.intersection(reserve_units)
        if constraint_i.empty:
            return
        r_t = get_var(network, component, f"r_{reserve_name}_{direction}")
        r = linexpr((1, r_t)).sel({component: constraint_i})

        if component == "Generator":
            p_t = get_var(network, component, "p")
            p = linexpr((1, p_t)).sel({component: constraint_i})
        if component == "StorageUnit":
            p_t_dispatch = get_var(network, component, "p_dispatch")
            p_t_store = get_var(network, component, "p_store")
            p = (p_t_dispatch - p_t_store).sel({component: constraint_i})
        p_nom = components.loc[constraint_i, "p_nom"]
        if direction == "up":
            p_max_pu = get_switchable_as_dense(network, component, "p_max_pu")[
                constraint_i
            ]
            p_max = p_max_pu * p_nom
            p_max = xr.DataArray(
                p_max.values, coords={"snapshot": p_max.index, component: p_max.columns}
            )
            define_constraints(
                network,
                r + p,
                "<=",
                p_max,
                component,
                f"{reserve_name}_up_reserve_headroom_non_ext",
            )
        if direction == "down":
            p_min_pu = get_switchable_as_dense(network, component, "p_min_pu")[
                constraint_i
            ]
            p_min = p_min_pu * p_nom
            p_min = xr.DataArray(
                p_min.values, coords={"snapshot": p_min.index, component: p_min.columns}
            )
            committable_i = network.get_committable_i(component).intersection(
                constraint_i
            )
            non_commmitable_i = constraint_i.difference(committable_i)
            if not committable_i.empty:
                status_t = (
                    get_var(network, component, "status")
                    .rename({f"{component}-com": component})
                    .sel({component: committable_i})
                )
                p_min_comm = p_min.sel({component: committable_i}) * status_t
                lhs = (p - r).sel({component: committable_i}) - p_min_comm
                define_constraints(
                    network,
                    lhs,
                    ">=",
                    0,
                    component,
                    f"{reserve_name}_down_reserve_headroom_non_ext_comm",
                )
            if not non_commmitable_i.empty:
                lhs = (p - r).sel({component: non_commmitable_i})
                rhs = p_min.sel({component: non_commmitable_i})
                define_constraints(
                    network,
                    lhs,
                    ">=",
                    rhs,
                    component,
                    f"{reserve_name}_down_reserve_headroom_non_ext_non_comm",
                )

        if component == "Generator":
            # TODO: omit for committable units after status incorporated into ramping constraint
            # Linear approximation of the status = 1 constraint
            # r_max_pu = min(ramp_limit, p_max_pu - p_min_pu)
            r_max_pu = components.loc[
                constraint_i, f"{reserve_name}_reserve_ramp_limit_{direction}"
            ]
            p_max_pu = components.loc[constraint_i, "p_max_pu"]
            p_min_pu_stable = components.loc[constraint_i, "p_min_pu_stable"]
            operational_band = p_max_pu - p_min_pu_stable
            r_max_pu.loc[operational_band < r_max_pu] = operational_band.loc[
                operational_band < r_max_pu
            ]

            spinning_i = p_min_pu_stable[p_min_pu_stable > 0].index

            if direction == "up":
                max_r_over_p = r_max_pu[spinning_i] / p_min_pu_stable[spinning_i]
            elif direction == "down":
                max_r_over_p = 1 / (
                    p_min_pu_stable[spinning_i] / r_max_pu[spinning_i] + 1
                )
            max_r_over_p = max_r_over_p.rename_axis(component).to_xarray()

            minus_max_r = linexpr((-max_r_over_p, p_t.sel({component: spinning_i})))
            define_constraints(
                network,
                r + minus_max_r,
                "<=",
                0,
                component,
                f"{reserve_name}_{direction}_reserve_spinning_non_ext",
            )
        if component == "StorageUnit":
            soc_end_t = get_var(network, component, "state_of_charge").sel(
                {component: constraint_i}
            )
            # following based on https://github.com/PyPSA/PyPSA/blob/0555a5b4cc8c26995f2814927cf250e928825cba/pypsa/optimization/constraints.py#L731
            ps = network.investment_periods.rename("period")
            sl = slice(None, None)
            soc_name = soc_end_t.name
            soc_begin_t = [
                soc_end_t.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps
            ]
            soc_begin_t = xr.concat(soc_begin_t, dim="snapshot")
            soc_begin_t = linopy.variables.Variable(
                soc_begin_t, network.model
            )  # , soc_name)

            if direction == "up":
                # (dispatch power + reserve up power) * hours per timestep / dispatch efficiency <= stored energy
                eff = components.loc[constraint_i, "efficiency_dispatch"]
                eff = eff.rename_axis(component).to_xarray()
                minus_soc_times_eff_over_hours = linexpr(
                    (-eff / hours_per_timestep, soc_begin_t)
                )
                define_constraints(
                    network,
                    r + p + minus_soc_times_eff_over_hours,
                    "<=",
                    0,
                    component,
                    f"{reserve_name}_up_reserve_charge_constraint_non_ext",
                )
            if direction == "down":
                # (store power + reserve down power) * hours per timestep * store efficiency <= storage capacity - stored energy
                eff = components.loc[constraint_i, "efficiency_store"]
                eff = eff.rename_axis(component).to_xarray()
                soc_over_eff_over_hours = linexpr(
                    (1 / (eff * hours_per_timestep), soc_begin_t)
                )
                soc_max = (
                    components.loc[constraint_i, "p_nom"]
                    * components.loc[constraint_i, "max_hours"]
                )
                soc_max = soc_max.rename_axis(component).to_xarray()
                define_constraints(
                    network,
                    r - p + soc_over_eff_over_hours,
                    "<=",
                    soc_max / (eff * hours_per_timestep),
                    component,
                    f"{reserve_name}_down_reserve_charge_constraint_non_ext",
                )
        # TODO: incorporate status variable in this constraint for committable units
        ramp_limit = components.loc[
            constraint_i, f"{reserve_name}_reserve_ramp_limit_{direction}"
        ]
        ramp_limit_i = ramp_limit[ramp_limit < 1].index
        r_max = (p_nom * ramp_limit).rename_axis(component).to_xarray()
        define_constraints(
            network,
            r.sel({component: ramp_limit_i}),
            "<=",
            r_max.sel({component: ramp_limit_i}),
            component,
            f"{reserve_name}_{direction}_ramp_limit_non_ext",
        )

    return warm_reserve_non_ext


# WARNING: currently not functional
# TODO: adapt to the new warm reserve constraint formulation
def make_warm_reserve_ext_constraints(reserve_name="warm", direction="up"):
    def warm_reserve_ext(
        network, snapshots, component, reserve_units, max_r_over_p, hours_per_timestep
    ):
        components = (
            network.generators if component == "Generator" else network.storage_units
        )
        ext_i = network.get_extendable_i(component)
        constraint_i = ext_i.intersection(reserve_units)
        if constraint_i.empty:
            return
        r_t = get_var(network, component, f"r_{reserve_name}_{direction}").sel(
            {component: constraint_i}
        )
        p_t = get_var(
            network, component, "p" if component == "Generator" else "p_dispatch"
        ).sel({component: constraint_i})
        p_nom = get_var(network, component, "p_nom").sel({component: constraint_i})
        p_max_pu = get_switchable_as_dense(network, component, "p_max_pu")[constraint_i]
        p_max_pu = xr.DataArray(
            p_max_pu.values,
            coords={"snapshot": p_max_pu.index, component: p_max_pu.columns},
        )
        r = linexpr((1, r_t))
        p = linexpr((1, p_t))
        minus_p_max = linexpr((-p_max_pu, p_nom))
        define_constraints(
            network,
            r + p + minus_p_max,
            "<=",
            0,
            component,
            f"reserve_{reserve_name}_{direction}_headroom_ext",
        )
        if component == "Generator":
            minus_max_r = linexpr((-max_r_over_p, p_t))
            define_constraints(
                network,
                r + minus_max_r,
                "<=",
                0,
                component,
                f"symmetric_{reserve_name}_{direction}_reserves_ext",
            )
        if component == "StorageUnit":
            # https://github.com/PyPSA/PyPSA/blob/master/pypsa/linopf.py#L529
            # (dispatch power + reserve power) * hours per timestep / efficiency <= stored energy
            eff = components.loc[constraint_i, "efficiency_dispatch"]
            eff = eff.rename_axis(component).to_xarray()
            soc_end_t = get_var(network, component, "state_of_charge").sel(
                {component: constraint_i}
            )
            # following based on https://github.com/PyPSA/PyPSA/blob/0555a5b4cc8c26995f2814927cf250e928825cba/pypsa/optimization/constraints.py#L731
            ps = network.investment_periods.rename("period")
            sl = slice(None, None)
            soc_name = soc_end_t.name
            soc_begin_t = [
                soc_end_t.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps
            ]
            soc_begin_t = xr.concat(soc_begin_t, dim="snapshot")
            soc_begin_t = linopy.variables.Variable(
                soc_begin_t, network.model
            )  # , soc_name)
            minus_soc_times_eff_over_hours = linexpr(
                (-eff / hours_per_timestep, soc_begin_t)
            )
            define_constraints(
                network,
                r + p + minus_soc_times_eff_over_hours,
                "<=",
                0,
                component,
                f"{reserve_name}_{direction}_reserve_charge_constraint_ext",
            )

    return warm_reserve_ext


# WARNING: currently functional for non extendable capacities and for one type of warm reserve only
# TODO: allow for different warm reserves at the same time
# TODO: implement the new warm reserve constraint for extendable capacities
def make_warm_reserve_constraints(
    reserve_name="warm",
    reserve_factor=1,
    directions=["up"],
):
    def warm_reserve(
        network,
        snapshots,
        warm_reserve_need_per_demand,
        warm_reserve_need_per_pv,
        warm_reserve_need_per_wind,
        max_r_over_p,
        hours_per_timestep,
    ):
        # https://github.com/PyPSA/pypsa-eur/blob/378d1ef82bf874d23e900b5933c16d5081fcfec9/scripts/solve_network.py#L319
        fixed_r_target = (
            get_fixed_demand(network, snapshots)
            * warm_reserve_need_per_demand
            * reserve_factor
        )
        minus_variable_r_target_list = []
        if warm_reserve_need_per_demand > 0:
            minus_variable_r_target_list.append(
                get_variable_demand(
                    network,
                    snapshots,
                    scale_factor=-warm_reserve_need_per_demand * reserve_factor,
                )
            )
        if warm_reserve_need_per_pv > 0:
            minus_variable_r_target_list.append(
                get_variable_supply(
                    network,
                    snapshots,
                    scale_factor=-warm_reserve_need_per_pv * reserve_factor,
                    carriers=["PV ground", "PV roof"],
                )
            )
        if warm_reserve_need_per_wind > 0:
            minus_variable_r_target_list.append(
                get_variable_supply(
                    network,
                    snapshots,
                    scale_factor=-warm_reserve_need_per_wind * reserve_factor,
                    carriers=["Wind onshore", "Wind offshore"],
                )
            )
        minus_variable_r_target = sum(minus_variable_r_target_list)

        for direction in directions:
            r_t_list = []
            for component in ["Generator", "StorageUnit"]:
                components = (
                    network.generators
                    if component == "Generator"
                    else network.storage_units
                )
                reserve_i = components[
                    components["bus"].str.startswith("PL")
                    & (components[f"is_{reserve_name}_reserve"] == True)
                ].index
                if reserve_i.empty:
                    continue

                define_variables(
                    network,
                    0,
                    np.inf,
                    component,
                    f"r_{reserve_name}_{direction}",
                    axes=[snapshots, reserve_i],
                )
                make_warm_reserve_ext_constraints(reserve_name, direction)(
                    network,
                    snapshots,
                    component,
                    reserve_units=reserve_i,
                    max_r_over_p=max_r_over_p,
                    hours_per_timestep=hours_per_timestep,
                )
                make_warm_reserve_non_ext_constraints(reserve_name, direction)(
                    network,
                    snapshots,
                    component,
                    reserve_units=reserve_i,
                    max_r_over_p=max_r_over_p,
                    hours_per_timestep=hours_per_timestep,
                )
                r_t = (
                    get_var(network, component, f"r_{reserve_name}_{direction}")
                    .sel({component: reserve_i})
                    .sum(component)
                )
                r_t_list.append(r_t)
            total_r = sum(r_t_list)
            define_constraints(
                network,
                total_r + minus_variable_r_target,
                ">=",
                fixed_r_target,
                "Generator+StorageUnit",
                f"total_{reserve_name}_{direction}_reserve",
            )

    return warm_reserve


def cold_reserve(
    network,
    snapshots,
    cold_reserve_need_per_demand,
    cold_reserve_need_per_import,
    warm_reserve_names=["warm"],
):
    fixed_r_target = get_fixed_demand(network, snapshots) * cold_reserve_need_per_demand
    minus_variable_r_target_list = []
    if cold_reserve_need_per_demand > 0:
        minus_variable_r_target_list.append(
            get_variable_demand(
                network, snapshots, scale_factor=-cold_reserve_need_per_demand
            )
        )
    if cold_reserve_need_per_import > 0:
        minus_variable_r_target_list.append(
            get_variable_supply(
                network,
                snapshots,
                scale_factor=-cold_reserve_need_per_import,
                carriers=["AC", "DC"],
            )
        )
    minus_variable_r_target = sum(minus_variable_r_target_list)

    variable_r_list = []
    fixed_r_list = []

    for component, components in [
        ("Generator", network.generators),
        ("Link", network.links),
    ]:
        # Links of JWCD category delivering power have negative p0

        reserve_i = components[
            components["area"].str.startswith("PL")
            & (components["is_cold_reserve"] == True)
        ].index

        # Non-extendable capacities
        non_ext_i = network.get_non_extendable_i(component)
        non_ext_reserve_i = non_ext_i.intersection(reserve_i)
        if not non_ext_reserve_i.empty:
            p_t = get_var(network, component, "p").sel({component: non_ext_reserve_i})
            p_nom = components.loc[non_ext_reserve_i, "p_nom"]
            p_max_pu = get_switchable_as_dense(
                network,
                component,
                "p_max_pu" if component == "Generator" else "p_min_pu",
            )[non_ext_reserve_i]
            minus_p = linexpr((-1, p_t)).sum(component)
            p_max = (p_max_pu * p_nom).sum(axis=1)
            p_max = xr.DataArray(p_max, dims="snapshot")
            if component == "Link":
                p_max = (-1) * p_max
                minus_p = (-1) * minus_p
            variable_r_list.append(minus_p)
            fixed_r_list.append(p_max)

        # Extendable capacities
        ext_i = network.get_extendable_i(component)
        ext_reserve_i = ext_i.intersection(reserve_i)
        if not ext_reserve_i.empty:
            p_t = get_var(network, component, "p").sel({component: ext_reserve_i})
            p_nom = (
                get_var(network, component, "p_nom")
                .sel({f"{component}-ext": ext_reserve_i})
                .rename({f"{component}-ext": component})
            )
            p_max_pu = get_switchable_as_dense(
                network,
                component,
                "p_max_pu" if component == "Generator" else "p_min_pu",
            )[ext_reserve_i]
            p_max_pu = xr.DataArray(
                p_max_pu.values,
                coords={"snapshot": p_max_pu.index, component: p_max_pu.columns},
            )
            minus_p = linexpr((-1, p_t)).sum(component)
            p_max = linexpr((p_max_pu, p_nom)).sum(component)
            if component == "Link":
                p_max = (-1) * p_max
                minus_p = (-1) * minus_p
            variable_r_list.append(p_max + minus_p)

    # Cold + warm reserve
    variable_r = sum(variable_r_list)
    fixed_r = sum(fixed_r_list)

    # Warm reserve
    if warm_reserve_names:
        r_t_list = []
        for reserve_name in warm_reserve_names:
            r_t_list.append(
                get_var(network, "Generator", f"r_{reserve_name}_up").sel(
                    {"Generator": reserve_i}
                )
            )
        minus_warm_r = -sum(r_t_list).sum("Generator")
        lhs = variable_r + minus_warm_r + minus_variable_r_target
    else:
        lhs = variable_r + minus_variable_r_target

    define_constraints(
        network,
        lhs,
        ">=",
        -fixed_r + fixed_r_target,
        "Generator+Link",
        "total_cold_reserve",
    )


def maximum_capacity_per_area(network, snapshots):
    # Custom reimplementation of the maximum capacity constraint using area attribute
    # Compared to PyPSA's default capacity constraint it
    # (1) works for links
    # (2) includes fixed capacities
    # https://github.com/PyPSA/PyPSA/blob/438532d956fce10382d1fac4c7fc9c2fa975496d/pypsa/optimization/global_constraints.py#L86

    model = network.model
    cols = network.areas.columns[network.areas.columns.str.startswith("nom_")]
    areas = network.areas.index[network.areas[cols].notnull().any(axis=1)].rename(
        "Area-nom_min_max"
    )

    for col in cols:
        msg = (
            f"Bus column '{col}' has invalid specification and cannot be "
            "interpreted as constraint, must match the pattern "
            "`nom_{min/max}_{carrier}_{period}`"
        )
        if col.startswith("nom_min_"):
            sign = ">="
        elif col.startswith("nom_max_"):
            sign = "<="
        else:
            logging.warning(msg)
            continue
        remainder = col[len("nom_max_") :]
        carrier, period = remainder.rsplit("_", 1)
        period = int(period)
        if carrier not in network.carriers.index or period not in snapshots.unique(
            "period"
        ):
            logging.warning(msg)
            continue

        lhs = []
        rhs = network.areas.loc[areas, col]

        for c, attr in [
            ("Generator", "p_nom"),
            ("Link", "p_nom"),
            ("StorageUnit", "p_nom"),
            ("Store", "e_nom"),
        ]:
            var = f"{c}-{attr}"
            dim = f"{c}-ext"
            components = network.df(c)

            # Fixed capacities
            non_ext_i = network.get_non_extendable_i(c).intersection(
                components.index[components["carrier"] == carrier]
            )
            non_ext_i = non_ext_i[network.get_active_assets(c, period)[non_ext_i]]
            # Group p_nom of components by area and sum
            nom_fixed = components.loc[non_ext_i].groupby("area")[attr].sum()
            rhs -= nom_fixed.reindex(rhs.index, fill_value=0)

            # Extendable capacities
            ext_i = (
                network.get_extendable_i(c)
                .intersection(components.index[components["carrier"] == carrier])
                .rename(dim)
            )
            ext_i = ext_i[network.get_active_assets(c, period)[ext_i]]

            if ext_i.empty:
                continue

            areamap = components.loc[ext_i, "area"].rename(areas.name).to_xarray()
            expr = (
                model[var]
                .loc[ext_i]
                .groupby(areamap)
                .sum()
                .reindex({areas.name: areas})
            )
            lhs.append(expr)

        if not lhs:
            continue

        lhs = linopy.expressions.merge(lhs)
        mask = rhs.notnull()
        network.model.add_constraints(lhs, sign, rhs, f"Area-{col}", mask=mask)


def maximum_snsp(network, snapshots, max_snsp, ns_carriers):
    fixed_max_ns_supply = get_fixed_demand(network, snapshots) * max_snsp
    minus_variable_max_ns_supply = get_variable_demand(
        network, snapshots, scale_factor=-max_snsp
    )
    ns_supply = get_variable_supply(network, snapshots, carriers=ns_carriers)
    ac_supply = get_variable_supply(
        network, snapshots, scale_factor=max_snsp, carriers=["AC"]
    )

    define_constraints(
        network,
        ns_supply + ac_supply + minus_variable_max_ns_supply,
        "<=",
        fixed_max_ns_supply,
        "Generator+StorageUnit+Link",
        "maximum_snsp",
    )


def bev_ext_constraint(network, snapshots):
    bevs_ext = network.links[
        (network.links["carrier"] == "BEV") & network.links["p_nom_extendable"]
    ]
    bevs_ext_i = bevs_ext.index
    if bevs_ext.empty:
        return

    # Reference p_nom / e_nom values are needed to calculate capacity ratios
    # TODO: get ratios directly from technology data
    assert (bevs_ext["p_nom"] > 0).all()

    p_nom_bev = get_var(network, "Link", "p_nom").sel({"Link-ext": bevs_ext_i})

    # Battery capacity constraint
    bev_batteries_ext_i = bevs_ext_i.str.replace("BEV", "BEV battery")
    bev_batteries_ext = network.stores.loc[bev_batteries_ext_i].copy()
    bev_batteries_ext.index = bevs_ext_i

    assert (bev_batteries_ext["e_nom"] > 0).all()
    battery_ratio = (
        (bev_batteries_ext["e_nom"] / bevs_ext["p_nom"])
        .rename_axis("Link-ext")
        .to_xarray()
    )

    e_nom_battery = (
        get_var(network, "Store", "e_nom")
        .sel({"Store-ext": bev_batteries_ext_i})
        .rename({"Store-ext": "Link-ext"})
    )
    e_nom_battery = e_nom_battery.assign_coords(
        {"Link-ext": e_nom_battery.coords["Link-ext"].str.replace("BEV battery", "BEV")}
    )
    define_constraints(
        network,
        e_nom_battery - battery_ratio * p_nom_bev,
        "==",
        0,
        "Link-ext",
        "bev_battery_capacity",
    )

    # Charger capacity constraint
    bev_chargers_ext_i = bevs_ext_i.str.replace("BEV", "BEV charger")
    bev_chargers_ext = network.links.loc[bev_chargers_ext_i].copy()
    bev_chargers_ext.index = bevs_ext_i

    assert (bev_chargers_ext["p_nom"] > 0).all()
    charger_ratio = (
        (bev_chargers_ext["p_nom"] / bevs_ext["p_nom"])
        .rename_axis("Link-ext")
        .to_xarray()
    )

    p_nom_charger = get_var(network, "Link", "p_nom").sel(
        {"Link-ext": bev_chargers_ext_i}
    )
    p_nom_charger = p_nom_charger.assign_coords(
        {"Link-ext": p_nom_charger.coords["Link-ext"].str.replace("BEV charger", "BEV")}
    )
    define_constraints(
        network,
        p_nom_charger - charger_ratio * p_nom_bev,
        "==",
        0,
        "Link-ext",
        "bev_charger_capacity",
    )

    # V2G capacity constraint
    if "BEV V2G" not in network.links["carrier"].unique():
        return

    bev_v2g_ext_i = bevs_ext_i.str.replace("BEV", "BEV V2G")
    bev_v2g_ext = network.links.loc[bev_v2g_ext_i].copy()
    bev_v2g_ext.index = bevs_ext_i

    assert (bev_v2g_ext["p_nom"] > 0).all()
    v2g_ratio = (
        (bev_v2g_ext["p_nom"] / bevs_ext["p_nom"]).rename_axis("Link-ext").to_xarray()
    )

    p_nom_v2g = get_var(network, "Link", "p_nom").sel({"Link-ext": bev_v2g_ext_i})
    p_nom_v2g = p_nom_v2g.assign_coords(
        {"Link-ext": p_nom_v2g.coords["Link-ext"].str.replace("BEV V2G", "BEV")}
    )
    define_constraints(
        network,
        p_nom_v2g - v2g_ratio * p_nom_bev,
        "==",
        0,
        "Link-ext",
        "bev_v2g_capacity",
    )


def bev_charge_constraint(network, snapshots, hour=6, charge_level=0.75):
    bev_batteries = network.stores[network.stores["carrier"] == "BEV battery"]
    bev_batteries_i = bev_batteries.index
    if bev_batteries_i.empty:
        return

    constraint_snapshots_i = network.snapshots[
        network.snapshots.get_level_values(1).hour == hour
    ]
    assert len(constraint_snapshots_i) == 365

    e_t = (
        get_var(network, "Store", "e")
        .sel({"Store": bev_batteries_i})
        .rename({"Store": "Store-BEV_battery"})
    )
    e_t = e_t.sel({"snapshot": constraint_snapshots_i})

    e_nom = network.stores.loc[bev_batteries_i, "e_nom"]
    assert (e_nom > 0).all()
    e_nom = e_nom.rename_axis("Store-BEV_battery").to_xarray()

    define_constraints(
        network,
        e_t / e_nom,
        ">=",
        charge_level,
        "Store-BEV_battery",
        "bev_battery_min_level_cyclic",
    )


def chp_ext_constraint(network, snapshots):
    for component, components in [
        ("Generator", network.generators),
        ("Link", network.links),
    ]:
        chp_elec_ext = components[
            components["technology"].fillna("").str.contains("CHP")
            & ~components["technology"].fillna("").str.contains("heat output")
            & components["p_nom_extendable"]
        ]
        if component == "Link":
            # Assume actual heat output is defined at link's input
            assert (chp_elec_ext["p0_sign"] < 0).all()

        chp_elec_ext_i = chp_elec_ext.index
        if chp_elec_ext.empty:
            return

        chp_heat_ext_i = chp_elec_ext_i.str.replace("CHP", "CHP heat output")
        chp_heat_ext = network.generators.loc[chp_heat_ext_i].copy()
        chp_heat_ext.index = chp_elec_ext_i

        p_nom_elec = get_var(network, component, "p_nom").sel(
            {f"{component}-ext": chp_elec_ext_i}
        )

        p_nom_heat = get_var(network, "Generator", "p_nom").sel(
            {"Generator-ext": chp_heat_ext_i}
        )
        if component != "Generator":
            p_nom_heat = p_nom_heat.rename({"Generator-ext": f"{component}-ext"})

        p_nom_heat = p_nom_heat.assign_coords(
            {
                f"{component}-ext": p_nom_heat.coords[f"{component}-ext"].str.replace(
                    "CHP heat output", "CHP"
                )
            }
        )

        # Net thermal efficiency is always equal to heat output generator's efficiency
        # If the primary CHP component is a link, net electrical efficiency is an inverse of the link's efficiency
        elec_efficiency = chp_elec_ext["efficiency"]
        if component == "Link":
            elec_efficiency = 1 / elec_efficiency
        heat_to_elec_ratio = (
            (chp_heat_ext["efficiency"] / elec_efficiency)
            .rename_axis(f"{component}-ext")
            .to_xarray()
        )

        define_constraints(
            network,
            p_nom_heat - heat_to_elec_ratio * p_nom_elec,
            "==",
            0,
            f"{component}-ext",
            f"chp_heat_output_capacity",
        )


def chp_dispatch_constraint(network, snapshots):
    for component, components in [
        ("Generator", network.generators),
        ("Link", network.links),
    ]:
        chp_elec_ext = components[
            components["technology"].fillna("").str.contains("CHP")
            & ~components["technology"].fillna("").str.contains("heat output")
        ]
        if component == "Link":
            # Assume actual heat output is defined at link's input
            assert (chp_elec_ext["p0_sign"] < 0).all()

        chp_elec_ext_i = chp_elec_ext.index
        if chp_elec_ext.empty:
            return

        chp_heat_ext_i = chp_elec_ext_i.str.replace("CHP", "CHP heat output")
        chp_heat_ext = network.generators.loc[chp_heat_ext_i].copy()
        chp_heat_ext.index = chp_elec_ext_i

        p_elec = get_var(network, component, "p").sel({component: chp_elec_ext_i})

        p_heat = get_var(network, "Generator", "p").sel({"Generator": chp_heat_ext_i})
        if component != "Generator":
            p_heat = p_heat.rename({"Generator": component})

        p_heat = p_heat.assign_coords(
            {component: p_heat.coords[component].str.replace("CHP heat output", "CHP")}
        )

        # Net thermal efficiency is always equal to heat output generator's efficiency
        # If the primary CHP component is a link, net electrical efficiency is an inverse of the link's efficiency
        # Additionally p0_sign is negative for links so multiply by -1
        elec_efficiency = chp_elec_ext["efficiency"]
        if component == "Link":
            elec_efficiency = -1 / elec_efficiency
        heat_to_elec_ratio = (
            (chp_heat_ext["efficiency"] / elec_efficiency)
            .rename_axis(component)
            .to_xarray()
        )
        lhs = p_heat - heat_to_elec_ratio * p_elec
        lhs = lhs.rename({component: f"{component}-chp"})

        define_constraints(
            network,
            lhs,
            "==",
            0,
            f"{component}-chp",
            f"chp_heat_output_dispatch",
        )


def maximum_resistive_heater_small_production(
    network, snapshots, resistive_to_heat_pump_max_ratio
):
    resistive_heater_i = network.links[
        network.links["technology"] == "Resistive heater small"
    ].index
    heat_pump_i = network.links[network.links["technology"] == "Heat pump small"].index

    assert (network.links.loc[resistive_heater_i, "p0_sign"] < 0).all()
    assert (network.links.loc[heat_pump_i, "p0_sign"] < 0).all()

    p_t_resistive_heater = get_var(network, "Link", "p").sel(
        {"Link": resistive_heater_i}
    )

    p_t_heat_pump = get_var(network, "Link", "p").sel({"Link": heat_pump_i})

    p_annual_resistive_heater = linexpr((-1, p_t_resistive_heater)).sum("snapshot")
    p_annual_heat_pump = linexpr((-1, p_t_heat_pump)).sum("snapshot")

    define_constraints(
        network,
        p_annual_resistive_heater
        - resistive_to_heat_pump_max_ratio * p_annual_heat_pump,
        "<=",
        0,
        "Link",
        "maximum_resistive_heater_small_production",
    )


def store_dispatch_constraint_ext(network, snapshots):
    ext_i = network.get_extendable_i("StorageUnit")
    if ext_i.empty:
        return
    var_name = "StorageUnit-ext-store_dispatch"
    dispatch_t = (
        get_var(network, "StorageUnit", "p_dispatch")
        .rename({"StorageUnit": var_name})
        .sel({var_name: ext_i})
    )
    store_t = (
        get_var(network, "StorageUnit", "p_store")
        .rename({"StorageUnit": var_name})
        .sel({var_name: ext_i})
    )
    p_nom = (
        get_var(network, "StorageUnit", "p_nom")
        .rename({"StorageUnit-ext": var_name})
        .sel({var_name: ext_i})
    )
    define_constraints(
        network,
        dispatch_t + store_t - p_nom,
        "<=",
        0,
        var_name,
        "store_dispatch_constraint_ext",
    )


def store_dispatch_constraint_non_ext(network, snapshots):
    non_ext_i = network.get_non_extendable_i("StorageUnit")
    if non_ext_i.empty:
        return
    var_name = "StorageUnit-non_ext-store_dispatch"
    dispatch_t = (
        get_var(network, "StorageUnit", "p_dispatch")
        .rename({"StorageUnit": var_name})
        .sel({var_name: non_ext_i})
    )
    store_t = (
        get_var(network, "StorageUnit", "p_store")
        .rename({"StorageUnit": var_name})
        .sel({var_name: non_ext_i})
    )
    p_nom = (
        network.storage_units["p_nom"].loc[non_ext_i].rename_axis(var_name).to_xarray()
    )
    define_constraints(
        network,
        dispatch_t + store_t,
        "<=",
        p_nom,
        var_name,
        "store_dispatch_constraint_non_ext",
    )


def store_dispatch_constraint(network, snapshots):
    store_dispatch_constraint_ext(network, snapshots)
    store_dispatch_constraint_non_ext(network, snapshots)
