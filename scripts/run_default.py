from pypsa_pl.run_pypsa_pl import Params, run_pypsa_pl
from pypsa_pl.plot_results import (
    plot_generation,
    plot_capacity,
)
from pypsa_pl.io import product_dicts, dict_to_str


if __name__ == "__main__":
    param_space = {"year": [2025]}

    for run in product_dicts(**param_space):
        params = Params(
            scenario=f"default;{dict_to_str(run)}",
            years=[run["year"]],
            solver="highs",
        )
        run_pypsa_pl(params, use_cache=False, dry=False)

    for plot_function in [
        plot_generation,
        plot_capacity,
    ]:
        for sector in ["electricity", "heat", "light vehicles", "hydrogen"]:
            plot_function(
                "default",
                x_var="year",
                x_vals=param_space["year"],
                sector=sector,
                extension="png",
                force=True,
            )
