import os
import logging

from pypsa_pl.config import data_dir
from pypsa_pl.build_network import (
    load_and_preprocess_inputs,
    create_custom_network,
    add_snapshots,
    add_carriers,
    add_buses_and_areas,
    process_capacity_data,
    add_capacities,
    add_energy_flow_constraints,
    add_capacity_constraints,
)
from pypsa_pl.define_time_dependent_attributes import (
    define_time_dependent_attributes,
)
from pypsa_pl.optimise_network import optimise_network
from pypsa_pl.update_installed_capacity_data import update_installed_capacity_data

from pypsa_pl.plot_outputs import (
    plot_installed_capacities,
    plot_capacity_additions,
    plot_storage_capacities,
    plot_storage_capacity_additions,
    plot_annual_generation,
    plot_energy_balance_at_peak_load,
    plot_curtailed_vres_energy,
    plot_fuel_consumption,
    plot_co2_emissions,
    plot_detailed_costs,
    plot_total_costs,
    plot_capex,
)
from pypsa_pl.process_output_network import (
    calculate_fuel_consumption,
    calculate_co2_emissions,
    calculate_electricity_trade_revenue,
    calculate_sectoral_flows,
    calculate_sectoral_unit_costs,
    calculate_statistics,
)


def run_simulation(
    params,
    custom_input_operation=None,
    installed_capacity_variant=None,
    custom_installed_capacity_operation=None,
):

    run_dir = lambda *path: data_dir("runs", params["run_name"], *path)
    os.makedirs(run_dir(), exist_ok=True)

    inputs = load_and_preprocess_inputs(params, custom_operation=custom_input_operation)
    network = create_custom_network(params)

    add_snapshots(network, params)
    add_carriers(network, inputs, params)
    add_buses_and_areas(network, inputs, params)

    df_cap = process_capacity_data(inputs, params)
    df_attr_t = define_time_dependent_attributes(df_cap, params)

    add_capacities(network, df_cap, df_attr_t, params)
    add_energy_flow_constraints(network, inputs, params)
    add_capacity_constraints(network, inputs, params)

    network.export_to_csv_folder(run_dir("input_network"))
    df_attr_t.to_csv(run_dir("input_network", "time_dependent_attributes.csv"))

    networks = optimise_network(network, params)

    if len(networks) == 1:
        networks[0].export_to_csv_folder(run_dir("output_network"))
        network = networks[0]
    elif len(networks) == 2:
        # In this case, the optimisation is repeated with fixed capacities, such that marginal prices are affected only by variable costs
        networks[0].export_to_csv_folder(run_dir("output_network_non_fixed"))
        networks[1].export_to_csv_folder(run_dir("output_network"))
        network = networks[1]

    if installed_capacity_variant is not None:
        df_cap_update = update_installed_capacity_data(
            network,
            variant=installed_capacity_variant,
            original_variant_list=params["installed_capacity"],
            force_original=True,
            custom_operation=custom_installed_capacity_operation,
        )
        installed_capacity_file = data_dir(
            "input", f"installed_capacity;variant={installed_capacity_variant}.csv"
        )
        logging.info(f"Updating {installed_capacity_file} with output capacities")
        df_cap_update.to_csv(installed_capacity_file, index=False)

    return network


def generate_outputs(network, output_plots_dir=None, output_data_dir=None):

    if output_plots_dir is not None:
        os.makedirs(output_plots_dir(), exist_ok=True)
    if output_data_dir is not None:
        os.makedirs(output_data_dir(), exist_ok=True)

    def plot_and_save(plot_func, name, **kwargs):
        fig, df = plot_func(network, **kwargs)
        if output_plots_dir is not None:
            fig.savefig(output_plots_dir(f"{name}.png"))
        if output_data_dir is not None:
            df.to_csv(output_data_dir(f"{name}.csv"), index=False)

    def get_data_and_save(data_func, name, digits=3, **kwargs):
        df = data_func(network, **kwargs)
        if "value" in df.columns:
            df["value"] = df["value"].round(digits)
        if output_data_dir is not None:
            df.to_csv(output_data_dir(f"{name}.csv"), index=False)

    for plot_function in [
        plot_installed_capacities,
        plot_capacity_additions,
        plot_annual_generation,
    ]:
        for sector, bus_carriers in [
            ("electricity", ["electricity in", "electricity out"]),
            ("heat centralised", ["heat centralised in", "heat centralised out"]),
            ("heat decentralised", ["heat decentralised"]),
            ("light vehicle mobility", ["light vehicle mobility"]),
            ("hydrogen", ["hydrogen"]),
            ("natural gas", ["natural gas"]),
        ]:
            name = f"{plot_function.__name__[len('plot_') :]};sector={sector}"
            plot_and_save(
                plot_function,
                name,
                bus_carriers=bus_carriers,
                carrier_name=sector,
                make_fig=False,
            )

    for plot_function in [
        plot_annual_generation,
    ]:
        for sector, bus_carriers in [
            ("heat", ["space heating", "water heating", "other heating"]),
        ]:
            name = f"{plot_function.__name__[len('plot_') :]};sector={sector}"
            plot_and_save(
                plot_function,
                name,
                bus_carriers=bus_carriers,
                carrier_name=sector,
                make_fig=False,
            )

    for plot_function in [
        plot_energy_balance_at_peak_load,
    ]:
        for sector, bus_carriers, load_types in [
            (
                "electricity",
                ["electricity in", "electricity out"],
                ["residual", "vRES"],
            ),
            (
                "heat centralised",
                ["heat centralised in", "heat centralised out"],
                ["heating"],
            ),
            (
                "heat decentralised",
                ["heat decentralised"],
                ["heating"],
            ),
            (
                "natural gas",
                ["natural gas"],
                ["total"],
            ),
        ]:
            for load_type in load_types:
                name = f"{plot_function.__name__[len('plot_') :]};sector={sector};load_type={load_type}"
                plot_and_save(
                    plot_function,
                    name,
                    bus_carriers=bus_carriers,
                    carrier_name=sector,
                    load_type=load_type,
                    make_fig=False,
                )

    for plot_function in [
        plot_storage_capacities,
        plot_storage_capacity_additions,
    ]:
        for sector, bus_carriers in [
            ("electricity", ["hydro PSH electricity", "battery large electricity"]),
            ("heat centralised", ["heat storage large tank heat"]),
            ("heat decentralised", ["heat storage small heat"]),
            ("hydrogen", ["hydrogen"]),
        ]:
            name = f"{plot_function.__name__[len('plot_') :]};sector={sector}"
            plot_and_save(
                plot_function,
                name,
                bus_carriers=bus_carriers,
                carrier_name=sector,
                make_fig=False,
            )

    for plot_function in [
        plot_curtailed_vres_energy,
        plot_fuel_consumption,
        plot_co2_emissions,
        plot_detailed_costs,
        plot_total_costs,
    ]:
        name = plot_function.__name__[len("plot_") :]
        plot_and_save(plot_function, name, make_fig=False)

    for plot_function in [
        plot_capex,
    ]:
        for cost_attr in ["investment_cost"]:
            name = f"{plot_function.__name__[len('plot_') :]};cost_attr={cost_attr}"
            plot_and_save(plot_function, name, cost_attr=cost_attr, make_fig=False)

    for data_function in [
        calculate_fuel_consumption,
        calculate_co2_emissions,
        calculate_electricity_trade_revenue,
        calculate_sectoral_flows,
        calculate_sectoral_unit_costs,
        calculate_statistics,
    ]:
        name = f"{data_function.__name__[len('calculate_') :]}_detailed"
        get_data_and_save(data_function, name)
