from pypsa_pl.run_pypsa_pl import Params, run_pypsa_pl
from pypsa_pl.plot_results import (
    plot_production,
    plot_capacity,
    plot_capacity_utilisation,
    plot_reserve_by_technology,
    plot_reserve_margin,
    plot_curtailment,
    plot_fuel_consumption,
    plot_co2_emissions,
)
from pypsa_pl.io import product_dict, dict_to_str


if __name__ == "__main__":
    param_space = {"year": [2020, 2025]}

    for run in product_dict(**param_space):
        params = Params(
            scenario=f"pypsa_pl_v2;{dict_to_str(run)}",
            years=[run["year"]],
            solver="gurobi",
        )
        try:
            run_pypsa_pl(params, use_cache=False, dry=False)
        except Exception as e:
            print(f"Error in {params['scenario']}: {e}")

    for plot_function in [
        plot_production,
        plot_capacity,
        plot_capacity_utilisation,
        plot_reserve_by_technology,
        plot_reserve_margin,
        plot_curtailment,
        plot_fuel_consumption,
        plot_co2_emissions,
    ]:
        plot_function(
            "pypsa_pl_v2", extra_params=param_space, extension="png", force=True
        )
