import pandas as pd
import numpy as np

from pypsa.linopt import (
    get_var,
    linexpr,
    join_exprs,
    define_constraints,
    define_variables,
)
from pypsa.descriptors import get_switchable_as_dense


# global constraints https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#global-constraints
# custom constraints https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#custom-constraints-and-other-functionality
# new constraints https://pypsa.readthedocs.io/en/latest/examples/optimization-with-linopy.html
# max_growth per carrier https://pypsa.readthedocs.io/en/latest/components.html#carrier


def p_set_constraint(network, snapshots):
    p_set = network.generators_t["p_set"]
    constraint_i = p_set.columns
    if constraint_i.empty:
        return
    p_t = get_var(network, "Generator", "p")[constraint_i]
    p = linexpr((1, p_t))
    define_constraints(
        network,
        p,
        "==",
        p_set,
        "Generator",
        "p_set_constraint",
    )


def maximum_annual_capacity_factor_ext(network, snapshots):
    ext_i = network.get_extendable_i("Generator")
    constraint_i = ext_i.intersection(
        network.generators[network.generators["p_max_pu_annual"] < 1].index
    )
    if constraint_i.empty:
        return
    p_t = get_var(network, "Generator", "p")[constraint_i]
    p_max_pu_annual = network.generators["p_max_pu_annual"][constraint_i]
    p_nom = get_var(network, "Generator", "p_nom")[constraint_i]
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).apply(join_exprs)
    minus_p_max_annual = linexpr((-N * p_max_pu_annual, p_nom))
    # lhs = pd.concat([p_annual, minus_p_max_annual], axis=1).apply(join_exprs, axis=1)
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
    p_t = get_var(network, "Generator", "p")[constraint_i]
    p_max_pu_annual = network.generators["p_max_pu_annual"][constraint_i]
    p_nom = network.generators["p_nom"][constraint_i]
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).apply(join_exprs)
    define_constraints(
        network,
        p_annual,
        "<=",
        N * p_max_pu_annual * p_nom,
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
    p_t = get_var(network, "Generator", "p")[constraint_i]
    p_min_pu_annual = network.generators["p_min_pu_annual"][constraint_i]
    p_nom = get_var(network, "Generator", "p_nom")[constraint_i]
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).apply(join_exprs)
    minus_p_min_annual = linexpr((-N * p_min_pu_annual, p_nom))
    # lhs = pd.concat([p_annual, minus_p_min_annual], axis=1).apply(join_exprs, axis=1)
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
    p_t = get_var(network, "Generator", "p")[constraint_i]
    p_min_pu_annual = network.generators["p_min_pu_annual"][constraint_i]
    p_nom = network.generators["p_nom"][constraint_i]
    N = len(snapshots)
    p_annual = linexpr((1, p_t)).apply(join_exprs)
    define_constraints(
        network,
        p_annual,
        ">=",
        N * p_min_pu_annual * p_nom,
        "Generator",
        "minimum_annual_capacity_factor_non_ext",
    )


def minimum_annual_capacity_factor(network, snapshots):
    minimum_annual_capacity_factor_ext(network, snapshots)
    minimum_annual_capacity_factor_non_ext(network, snapshots)


def operational_reserve_non_ext(
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
    r_t = get_var(network, component, "r")[constraint_i]
    p_t = get_var(
        network, component, "p" if component == "Generator" else "p_dispatch"
    )[constraint_i]
    p_nom = components["p_nom"][constraint_i]
    p_max_pu = get_switchable_as_dense(network, component, "p_max_pu")[constraint_i]
    r = linexpr((1, r_t))
    p = linexpr((1, p_t))
    p_max = p_max_pu * p_nom
    define_constraints(
        network,
        r + p,
        "<=",
        p_max,
        component,
        "reserve_headroom_non_ext",
    )
    if component == "Generator":
        minus_max_r = linexpr((-max_r_over_p, p_t))
        define_constraints(
            network,
            r + minus_max_r,
            "<=",
            0,
            component,
            "symmetric_reserves_non_ext",
        )
    if component == "StorageUnit":
        # https://github.com/PyPSA/PyPSA/blob/master/pypsa/linopf.py#L529
        # (dispatch power + reserve power) * hours per timestep / efficiency <= stored energy
        eff = components["efficiency_dispatch"][constraint_i]
        soc_end_t = get_var(network, component, "state_of_charge")[constraint_i]
        soc_begin_t = soc_end_t.groupby(level=0, group_keys=False).apply(
            lambda df: pd.DataFrame(
                np.roll(df, 1, axis=0), index=df.index, columns=df.columns
            )
        )
        minus_soc_times_eff_over_hours = linexpr(
            (-eff / hours_per_timestep, soc_begin_t)
        )
        define_constraints(
            network,
            r + p + minus_soc_times_eff_over_hours,
            "<=",
            0,
            component,
            "reserve_charge_constraint_non_ext",
        )


def operational_reserve_ext(
    network, snapshots, component, reserve_units, max_r_over_p, hours_per_timestep
):
    components = (
        network.generators if component == "Generator" else network.storage_units
    )
    ext_i = network.get_extendable_i(component)
    constraint_i = ext_i.intersection(reserve_units)
    if constraint_i.empty:
        return
    r_t = get_var(network, component, "r")[constraint_i]
    p_t = get_var(
        network, component, "p" if component == "Generator" else "p_dispatch"
    )[constraint_i]
    p_nom = get_var(network, component, "p_nom")[constraint_i]
    p_max_pu = get_switchable_as_dense(network, component, "p_max_pu")[constraint_i]
    r = linexpr((1, r_t))
    p = linexpr((1, p_t))
    minus_p_max = linexpr((-p_max_pu, p_nom))
    define_constraints(
        network,
        r + p + minus_p_max,
        "<=",
        0,
        component,
        "reserve_headroom_ext",
    )
    if component == "Generator":
        minus_max_r = linexpr((-max_r_over_p, p_t))
        define_constraints(
            network,
            r + minus_max_r,
            "<=",
            0,
            component,
            "symmetric_reserves_ext",
        )
    if component == "StorageUnit":
        # https://github.com/PyPSA/PyPSA/blob/master/pypsa/linopf.py#L529
        # (dispatch power + reserve power) * hours per timestep / efficiency <= stored energy
        eff = components["efficiency_dispatch"][constraint_i]
        soc_end_t = get_var(network, component, "state_of_charge")[constraint_i]
        soc_begin_t = soc_end_t.groupby(level=0, group_keys=False).apply(
            lambda df: pd.DataFrame(
                np.roll(df, 1, axis=0), index=df.index, columns=df.columns
            )
        )
        minus_soc_times_eff_over_hours = linexpr(
            (-eff / hours_per_timestep, soc_begin_t)
        )
        define_constraints(
            network,
            r + p + minus_soc_times_eff_over_hours,
            "<=",
            0,
            component,
            "reserve_charge_constraint_ext",
        )


def operational_reserve(
    network, snapshots, reserve_margin, max_r_over_p, hours_per_timestep
):
    # https://github.com/PyPSA/pypsa-eur/blob/378d1ef82bf874d23e900b5933c16d5081fcfec9/scripts/solve_network.py#L319
    loads = network.loads_t["p_set"]
    loads = loads[[name for name in loads.columns if "PL" in name]]
    total_r_target = loads.sum(axis=1) * reserve_margin
    r_t_list = []
    for component in ["Generator", "StorageUnit"]:
        components = (
            network.generators if component == "Generator" else network.storage_units
        )
        reserve_i = components[
            components["bus"].str.startswith("PL") & (components["is_reserve"] == True)
        ].index
        if reserve_i.empty:
            continue
        define_variables(
            network,
            0,
            np.inf,
            component,
            "r",
            axes=[snapshots, reserve_i],
        )
        operational_reserve_ext(
            network,
            snapshots,
            component,
            reserve_units=reserve_i,
            max_r_over_p=max_r_over_p,
            hours_per_timestep=hours_per_timestep,
        )
        operational_reserve_non_ext(
            network,
            snapshots,
            component,
            reserve_units=reserve_i,
            max_r_over_p=max_r_over_p,
            hours_per_timestep=hours_per_timestep,
        )
        r_t = get_var(network, component, "r")[reserve_i]
        r_t_list.append(r_t)
    r_t = pd.concat(r_t_list, axis=1)
    total_r = linexpr((1, r_t)).apply(join_exprs, axis=1)
    define_constraints(
        network,
        total_r,
        ">=",
        total_r_target,
        "Generator+StorageUnit",
        "total_operational_reserve",
    )


def maximum_capacity_per_voivodeship(network, snapshots):
    ext_i = network.get_extendable_i("Generator")
    constraint_i = ext_i.intersection(
        network.generators[
            (network.generators["p_nom_min"] < network.generators["p_nom_max"])
            & network.generators["bus"].str.startswith("PL")
        ].index
    )
    if constraint_i.empty:
        return
    p_nom = get_var(network, "Generator", "p_nom")[constraint_i]
    p_nom_per_carrier_voivodeship = (
        linexpr((1, p_nom))
        .groupby(
            [
                network.generators["carrier"][constraint_i],
                network.generators["voivodeship"][constraint_i],
            ]
        )
        .apply(join_exprs)
    )
    df = network.voivodeships.reset_index().melt(
        id_vars="voivodeship", var_name="carrier_period", value_name="p_nom_max"
    )
    df = df[df["p_nom_max"] < np.inf]
    df["carrier"] = df["carrier_period"].str[8:-5]
    df["period"] = df["carrier_period"].str[-4:].astype(int)
    # TODO: implement for multi-period optimization
    df = df[df["period"] == df["period"].max()]
    p_nom_max = df.set_index(["carrier", "voivodeship"])["p_nom_max"]
    carrier_voivodeship_i = p_nom_per_carrier_voivodeship.index.intersection(
        p_nom_max.index
    )
    define_constraints(
        network,
        p_nom_per_carrier_voivodeship[carrier_voivodeship_i],
        "<=",
        p_nom_max[carrier_voivodeship_i],
        "Generator",
        "maximum_capacity_per_voivodeship",
    )


def maximum_growth_per_carrier(network, snapshots):
    ext_i = network.get_extendable_i("Generator")
    constraint_i = ext_i.intersection(
        network.generators[
            (network.generators["p_nom_min"] < network.generators["p_nom_max"])
            & network.generators["bus"].str.startswith("PL")
        ].index
    )
    if constraint_i.empty:
        return
    p_nom = get_var(network, "Generator", "p_nom")[constraint_i]
    p_nom_start = network.generators["p_nom_min"][constraint_i]
    p_nom_per_carrier = (
        linexpr((1, p_nom))
        .groupby(network.generators["carrier"][constraint_i])
        .apply(join_exprs)
    )
    p_nom_start_per_carrier = p_nom_start.groupby(
        network.generators["carrier"][constraint_i]
    ).sum()
    max_growth = network.carriers["max_growth"]
    max_growth = max_growth[max_growth < np.inf]
    carrier_i = p_nom_per_carrier.index.intersection(max_growth.index)
    define_constraints(
        network,
        p_nom_per_carrier[carrier_i],
        "<=",
        p_nom_start_per_carrier[carrier_i] + max_growth[carrier_i],
        "Generator",
        "maximum_growth_per_carrier",
    )
