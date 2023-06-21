import pandas as pd
import numpy as np
import xarray as xr

from pypsa.optimization.compat import (
    get_var,
    linexpr,
    define_constraints,
    define_variables,
)
import linopy
from pypsa.descriptors import get_switchable_as_dense


# global constraints https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#global-constraints
# custom constraints https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#custom-constraints-and-other-functionality
# new constraints https://pypsa.readthedocs.io/en/latest/examples/optimization-with-linopy.html
# max_growth per carrier https://pypsa.readthedocs.io/en/latest/components.html#carrier


def get_fixed_demand(network, snapshots):
    """Get fixed demand for each snapshot."""

    loads = network.loads_t["p_set"]
    loads = loads[[name for name in loads.columns if name.startswith("PL")]]
    demand_t = loads.sum(axis=1)
    demand_t = xr.DataArray(demand_t, dims=["snapshot"])
    return demand_t


def get_variable_demand(network, snapshots, scale_factor=1):
    """Get variable demand for each snapshot."""
    demand_t_list = []

    # Demand from charging storage units
    components = network.storage_units
    store_i = components[components["bus"].str.startswith("PL")].index
    if not store_i.empty:
        store_t = (
            get_var(network, "StorageUnit", "p_dispatch")
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

    # Aggregate variable demand
    demand_t = scale_factor * sum(demand_t_list)
    return demand_t


def get_variable_supply(network, snapshots, scale_factor=1, carriers=None):
    """Get variable supply for each snapshot."""
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

    # Aggregate variable supply
    supply_t = scale_factor * sum(supply_t_list)
    return supply_t


# def p_set_constraint(network, snapshots):
#     p_set = network.generators_t["p_set"]
#     constraint_i = p_set.columns
#     if constraint_i.empty:
#         return
#     p_t = get_var(network, "Generator", "p")[constraint_i]
#     p = linexpr((1, p_t))
#     define_constraints(
#         network,
#         p,
#         "==",
#         p_set,
#         "Generator",
#         "p_set_constraint",
#     )


def maximum_annual_capacity_factor_ext(network, snapshots):
    ext_i = network.get_extendable_i("Generator")
    constraint_i = ext_i.intersection(
        network.generators[network.generators["p_max_pu_annual"] < 1].index
    )
    if constraint_i.empty:
        return
    p_t = get_var(network, "Generator", "p").sel({"Generator": constraint_i})
    p_max_pu_annual = (
        network.generators.loc[constraint_i, "p_max_pu_annual"]
        .rename_axis("Generator")
        .to_xarray()
    )
    p_nom = get_var(network, "Generator", "p_nom").sel({"Generator": constraint_i})
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).sum("snapshot")
    minus_p_max_annual = linexpr((-N * p_max_pu_annual, p_nom))
    define_constraints(
        network,
        p_annual + minus_p_max_annual,
        "<=",
        0,
        "Generator",
        "maximum_annual_capacity_factor_ext",
    )


def maximum_annual_capacity_factor_non_ext(network, snapshots):
    non_ext_i = network.get_non_extendable_i("Generator")
    constraint_i = non_ext_i.intersection(
        network.generators[network.generators["p_max_pu_annual"] < 1].index
    )
    if constraint_i.empty:
        return
    p_t = get_var(network, "Generator", "p").sel({"Generator": constraint_i})
    p_max_pu_annual = network.generators.loc[constraint_i, "p_max_pu_annual"]
    p_nom = network.generators.loc[constraint_i, "p_nom"]
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).sum("snapshot")
    p_annual_max = (N * p_max_pu_annual * p_nom).rename_axis("Generator").to_xarray()
    define_constraints(
        network,
        p_annual,
        "<=",
        p_annual_max,
        "Generator",
        "maximum_annual_capacity_factor_non_ext",
    )


def maximum_annual_capacity_factor(network, snapshots):
    maximum_annual_capacity_factor_ext(network, snapshots)
    maximum_annual_capacity_factor_non_ext(network, snapshots)


def minimum_annual_capacity_factor_ext(network, snapshots):
    ext_i = network.get_extendable_i("Generator")
    constraint_i = ext_i.intersection(
        network.generators[network.generators["p_min_pu_annual"] > 0].index
    )
    if constraint_i.empty:
        return
    p_t = get_var(network, "Generator", "p").sel({"Generator": constraint_i})
    p_min_pu_annual = (
        network.generators.loc[constraint_i, "p_min_pu_annual"]
        .rename_axis("Generator")
        .to_xarray()
    )
    p_nom = get_var(network, "Generator", "p_nom").sel({"Generator": constraint_i})
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).sum("snapshot")
    minus_p_min_annual = linexpr((-N * p_min_pu_annual, p_nom))
    define_constraints(
        network,
        p_annual + minus_p_min_annual,
        ">=",
        0,
        "Generator",
        "minimum_annual_capacity_factor_ext",
    )


def minimum_annual_capacity_factor_non_ext(network, snapshots):
    non_ext_i = network.get_non_extendable_i("Generator")
    constraint_i = non_ext_i.intersection(
        network.generators[network.generators["p_min_pu_annual"] > 0].index
    )
    if constraint_i.empty:
        return
    p_t = get_var(network, "Generator", "p").sel({"Generator": constraint_i})
    p_min_pu_annual = network.generators.loc[constraint_i, "p_min_pu_annual"]
    p_nom = network.generators.loc[constraint_i, "p_nom"]
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).sum("snapshot")
    p_annual_min = (N * p_min_pu_annual * p_nom).rename_axis("Generator").to_xarray()
    define_constraints(
        network,
        p_annual,
        ">=",
        p_annual_min,
        "Generator",
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
            p = (p_t_dispatch + p_t_store).sel({component: constraint_i})
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
            soc_begin_t = [
                soc_end_t.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps
            ]
            soc_begin_t = xr.concat(soc_begin_t, dim="snapshot")
            soc_begin_t = linopy.variables.Variable(soc_begin_t, network.model)

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
            soc_begin_t = [
                soc_end_t.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps
            ]
            soc_begin_t = xr.concat(soc_begin_t, dim="snapshot")
            soc_begin_t = linopy.variables.Variable(soc_begin_t, network.model)
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

    components = network.generators
    reserve_i = components[
        components["bus"].str.startswith("PL") & (components["is_cold_reserve"] == True)
    ].index

    variable_r_list = []
    fixed_r_list = []

    # Non-extendable capacities
    non_ext_i = network.get_non_extendable_i("Generator")
    non_ext_reserve_i = non_ext_i.intersection(reserve_i)
    if not non_ext_i.empty:
        p_t = get_var(network, "Generator", "p").sel({"Generator": non_ext_reserve_i})
        p_nom = components.loc[non_ext_reserve_i, "p_nom"]
        p_max_pu = get_switchable_as_dense(network, "Generator", "p_max_pu")[
            non_ext_reserve_i
        ]
        minus_p = linexpr((-1, p_t)).sum("Generator")
        p_max = (p_max_pu * p_nom).sum(axis=1)
        p_max = xr.DataArray(p_max, dims="snapshot")
        variable_r_list.append(minus_p)
        fixed_r_list.append(p_max)

    # Extendable capacities
    ext_i = network.get_extendable_i("Generator")
    ext_reserve_i = ext_i.intersection(reserve_i)
    if not ext_i.empty:
        p_t = get_var(network, "Generator", "p").sel({"Generator": ext_reserve_i})
        p_nom = get_var(network, "Generator", "p_nom").sel({"Generator": ext_reserve_i})
        p_max_pu = get_switchable_as_dense(network, "Generator", "p_max_pu")[
            ext_reserve_i
        ]
        p_max_pu = xr.DataArray(
            p_max_pu.values,
            coord={"snapshot": p_max_pu.index, "Generator": p_max_pu.columns},
        )
        minus_p = linexpr((-1, p_t)).sum("Generator")
        p_max = linexpr((p_max_pu, p_nom)).sum("Generator")
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
        "Generator",
        "total_cold_reserve",
    )


def maximum_capacity_per_area(network, snapshots):
    # TODO: test implementation
    ext_i = network.get_extendable_i("Generator")
    constraint_i = ext_i.intersection(
        network.generators[
            (network.generators["p_nom_min"] < network.generators["p_nom_max"])
            & network.generators["bus"].str.startswith("PL")
        ].index
    )
    if constraint_i.empty:
        return
    p_nom = get_var(network, "Generator", "p_nom").sel({"Generator": constraint_i})
    carrier_area = (
        (
            network.generators.loc[constraint_i, "carrier"]
            + "_"
            + network.generators.loc[constraint_i, "area"]
        )
        .rename("carrier_area")
        .rename_axis("Generator")
        .to_xarray()
    )
    p_nom_per_carrier_area = p_nom.groupby_sum(carrier_area)
    df = network.area.reset_index().melt(
        id_vars="area", var_name="carrier_period", value_name="p_nom_max"
    )
    df = df[df["p_nom_max"] < np.inf]
    df["carrier"] = df["carrier_period"].str[8:-5]
    df["period"] = df["carrier_period"].str[-4:].astype(int)
    # TODO: implement for multi-period optimization
    df = df[df["period"] == df["period"].max()]
    df["carrier_area"] = df["carrier"] + "_" + df["area"]
    p_nom_max = df.set_index("carrier_area")["p_nom_max"]
    carrier_area_i = p_nom_per_carrier_area.indexes["carrier_area"].intersection(
        p_nom_max.index
    )
    p_nom_per_carrier_area = p_nom_per_carrier_area.sel(
        {"carrier_area": carrier_area_i}
    )
    p_nom_max = p_nom_max.rename_axis("carrier_area").to_xarray()
    define_constraints(
        network,
        p_nom_per_carrier_area,
        "<=",
        p_nom_max,
        "Generator",
        "maximum_capacity_per_area",
    )


def maximum_growth_per_carrier(network, snapshots):
    # TODO: test implementation
    ext_i = network.get_extendable_i("Generator")
    constraint_i = ext_i.intersection(
        network.generators[
            (network.generators["p_nom_min"] < network.generators["p_nom_max"])
            & network.generators["bus"].str.startswith("PL")
        ].index
    )
    if constraint_i.empty:
        return
    p_nom = get_var(network, "Generator", "p_nom").sel({"Generator": constraint_i})
    p_nom_start = network.generators.loc[constraint_i, "p_nom_min"]
    carrier = network.generators.loc[constraint_i, "carrier"]
    p_nom_per_carrier = p_nom.groupby_sum(carrier.rename_axis("Generator").to_xarray())
    p_nom_start_per_carrier = p_nom_start.groupby(carrier).sum()
    max_growth = network.carriers["max_growth"]
    max_growth = max_growth[max_growth < np.inf]
    carrier_i = p_nom_per_carrier.indexes["carrier"].intersection(max_growth.index)
    p_nom_per_carrier = p_nom_per_carrier.sel({"carrier": carrier_i})
    p_nom_per_carrier_max = (
        (p_nom_start_per_carrier[carrier_i] + max_growth[carrier_i])
        .rename_axis("carrier")
        .to_xarray()
    )
    define_constraints(
        network,
        p_nom_per_carrier,
        "<=",
        p_nom_per_carrier_max,
        "Generator",
        "maximum_growth_per_carrier",
    )


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
