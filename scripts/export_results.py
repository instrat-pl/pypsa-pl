import pandas as pd
import os


def create_emissions_data(network):
    # Set emisisons equivalent based on carrier to generators
    df = pd.merge(network.generators, network.carriers, left_on="carrier", right_index=True)
    emissions = network.generators_t.p * df["co2_emissions"] / df["efficiency"]

    return emissions


def create_generation_data(network):
    gen = network.generators_t.p

    gen = gen.groupby(gen.columns, axis=1).sum()

    gen = gen.groupby([gen.index.year, gen.index.month]).agg('sum')
    return gen


def create_srmc_data(network):
    srmc = network.buses_t.marginal_price
    srmc = srmc.groupby([srmc.index.year, srmc.index.month]).agg('mean')
    return srmc


def create_cbf_data(network):
    links_p0 = network.links_t.p0
    links_p0 = links_p0.groupby([links_p0.index.year, links_p0.index.month]).agg('mean')
    return links_p0


def export_network_results(scenario_name, year, network):
    save_dir = os.getcwd() + f'/pypsa/{scenario_name}/results/{year}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    emissions = create_emissions_data(network)
    generation = create_generation_data(network)
    srmc = create_srmc_data(network)
    import_power = create_cbf_data(network)

    network.export_to_netcdf(f'{save_dir}/network.nc')

    emissions.to_csv(f'{save_dir}/emissions.csv')
    generation.to_csv(f'{save_dir}/generation.csv')
    srmc.to_csv(f'{save_dir}/srmc.csv')
    import_power.to_csv(f'{save_dir}/cbf.csv')