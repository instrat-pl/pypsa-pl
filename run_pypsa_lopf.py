import pandas as pd
from pypsa.linopt import get_var, linexpr, define_constraints

import pypsa
import scripts.import_data_to_pypsa as base_data
from scripts.export_results import export_network_results
from scripts.create_date_index import create_date_index
from prepare_input_files import scenario_name, reference_year, scenario_assumptions_path

#  ------------- Network -------------------------------------------------------
network = pypsa.Network()
network.name = 'PyPSA-PL'
year = int(reference_year)

snapshots = create_date_index(year)

network.set_snapshots(snapshots)

# add buses
network.import_components_from_dataframe(base_data.buses, 'Bus')

# add lines
network.import_components_from_dataframe(base_data.lines, 'Line')
network.lines['s_nom_extendable'] = True  # It needs to be True due to simplyfying grid (simplified 220kV model and no 110kV network) while leaving normal load. Grid investments calculated outside of model.

# Cross border flow (cbf)
network.madd("Link", base_data.cbf_links_in.index,
             bus0=base_data.cbf_links_in['bus0'].tolist(), bus1=base_data.cbf_links_in['bus1'].tolist(),
             p_nom=base_data.cbf_links_in['p_nom'].tolist())

cbf_marginal_cost = pd.DataFrame(base_data.cbf_links_in)
cbf_marginal_cost = cbf_marginal_cost.drop_duplicates(subset=['bus0'])
cbf_marginal_cost.reset_index()
cbf_marginal_cost.set_index(keys=['bus0'])

network.madd("Generator", base_data.cbf_buses.index, carrier='CBF',
             bus=base_data.cbf_buses.index.tolist(), p_nom=20000, marginal_cost=cbf_marginal_cost['marginal_cost'].tolist()) # The marginal_cost on the CBF links determines the import volume. To decrease imports increase the marginal_cost or lower the p_nom of specific links in the cbf_links.xlsx file.

# Carriers - CO2 emissions from https://www.epa.gov/sites/production/files/2015-07/documents/emission-factors_2014.pdf and https://pypsa-eur.readthedocs.io/en/latest/costs.html

dict_carriers = {
    'Lignite': 0.334, 'Hard coal': 0.354, 'Gas': 0.187, 'Hydrogen': 0, 'Biomass': 0.403, 'Biogas': 0.178,
    'Geothermal': 0.026,
    'Wind': 0, 'Wind_Off': 0, 'PV': 0, 'hps': 0, 'Hydro': 0, 'CBF': 0, 'VOLL': 0, 'Battery': 0, 'Nuclear': 0
}

for carrier in dict_carriers:
    network.add("Carrier", name=carrier, co2_emissions=dict_carriers[carrier])

# Load
network.madd("Load", base_data.load.columns, bus=base_data.load.columns, p_set=base_data.load)

# Generators
# JWCD
network.madd("Generator", base_data.gen_jwcd.index, efficiency=base_data.gen_jwcd['efficiency'].to_list(),
             bus=base_data.gen_jwcd['bus'].tolist(), carrier=base_data.gen_jwcd['carrier'].to_list(),
             p_nom=base_data.gen_jwcd['p_nom'].tolist(), marginal_cost=base_data.gen_jwcd['marginal_cost'].to_list())
# nJWCD
network.madd("Generator", base_data.gen_njwcd.index, suffix='_nJWCD',
             efficiency=base_data.gen_njwcd['efficiency'].to_list(),
             bus=base_data.gen_njwcd['bus'].tolist(), carrier=base_data.gen_njwcd['carrier'].to_list(),
             p_nom=base_data.gen_njwcd['p_nom'].tolist())

# CHP
network.madd("Generator", base_data.gen_chp.index, suffix='_CHP',
             bus=base_data.gen_chp['bus'].tolist(), carrier=base_data.gen_chp['carrier'].to_list(),
             p_nom=base_data.gen_chp['p_nom'].tolist(), p_max_pu=base_data.gen_res_chp,
             efficiency=base_data.gen_chp['efficiency'].tolist())

# Wind Onshore
network.madd("Generator", base_data.gen_wind_onshore['bus'], suffix='_WIND',
             bus=base_data.gen_wind_onshore['bus'].tolist(), carrier="Wind",
             p_nom=base_data.gen_wind_onshore['p_nom'].tolist(), p_max_pu=base_data.gen_res_wind_onshore)

network.madd("Generator", base_data.gen_wind_onshore_new['bus'], suffix='_WIND_NEW',
             bus=base_data.gen_wind_onshore_new['bus'].tolist(), carrier="Wind",
             p_nom=base_data.gen_wind_onshore_new['p_nom'].tolist(), p_max_pu=base_data.gen_res_wind_onshore_new)

# Wind Offshore
network.madd("Generator", base_data.gen_wind_offshore.index,
             bus=base_data.gen_wind_offshore['bus'].tolist(), carrier='Wind_Off', suffix="_WIND_OFF",
             p_nom=base_data.gen_wind_offshore['p_nom'].tolist(), p_max_pu=base_data.gen_res_wind_offshore)

# pv
network.madd("Generator", base_data.gen_pv_ground['bus'], suffix='_PV_GROUND',
             bus=base_data.gen_pv_ground['bus'].tolist(), carrier=base_data.gen_pv_ground['carrier'].to_list(),
             p_nom=base_data.gen_pv_ground['p_nom'].tolist(), p_max_pu=base_data.gen_res_pv)

network.madd("Generator", base_data.gen_pv_rooftop['bus'], suffix='_PV_ROOF',
             bus=base_data.gen_pv_rooftop['bus'].tolist(), carrier=base_data.gen_pv_rooftop['carrier'].to_list(),
             p_nom=base_data.gen_pv_rooftop['p_nom'].tolist(), p_max_pu=base_data.gen_res_pv)

# DSR
network.madd("Generator", base_data.gen_dsr['bus'], suffix='_DSR', marginal_cost=1000,
             bus=base_data.gen_dsr['bus'].tolist(), carrier=base_data.gen_dsr['carrier'].tolist(),
             p_nom=base_data.gen_dsr['p_nom'].tolist())

# ind: hc
network.madd("Generator", base_data.gen_ind_hc['bus'], suffix='_IND_HC',
             bus=base_data.gen_ind_hc['bus'].tolist(), carrier=base_data.gen_ind_hc['carrier'].tolist(),
             p_nom=base_data.gen_ind_hc['p_nom'].tolist(), p_max_pu=0.9,
             efficiency=base_data.gen_ind_hc['efficiency'].tolist())

# ind: gas
network.madd("Generator", base_data.gen_ind_gas['bus'], suffix='_IND_GAS',
             bus=base_data.gen_ind_gas['bus'].tolist(), carrier=base_data.gen_ind_gas['carrier'].tolist(),
             p_nom=base_data.gen_ind_gas['p_nom'].tolist(), p_max_pu=1,
             efficiency=base_data.gen_ind_gas['efficiency'].tolist())

# biogas
network.madd("Generator", base_data.gen_bg['bus'], suffix='_BG',
             bus=base_data.gen_bg['bus'].tolist(), carrier="Biogas",
             p_nom=base_data.gen_bg['p_nom'].tolist(), p_max_pu=1, efficiency=base_data.gen_bg['efficiency'].tolist())

# biomass
network.madd("Generator", base_data.gen_bm['bus'], suffix='_BM',
             bus=base_data.gen_bm['bus'].tolist(), carrier="Biomass",
             p_nom=base_data.gen_bm['p_nom'].tolist(), p_max_pu=1, efficiency=base_data.gen_bm['efficiency'].tolist())

# geothermal
network.madd("Generator", base_data.gen_geo['bus'], suffix='_GEO',
             bus=base_data.gen_geo['bus'].tolist(), carrier="Geothermal",
             p_nom=base_data.gen_geo['p_nom'].tolist(), p_max_pu=base_data.gen_geo['p_max_pu'].tolist())

# ror
network.madd("Generator", base_data.gen_ror['bus'], suffix='_ROR',
             bus=base_data.gen_ror['bus'].tolist(), carrier=base_data.gen_ror['carrier'].tolist(),
             p_nom=base_data.gen_ror['p_nom'].tolist(), p_max_pu=base_data.gen_ror['p_max_pu'].tolist())

# Storage units
# HPS
network.madd("StorageUnit", base_data.gen_hps.index, suffix='_Battery',
             bus=base_data.gen_hps['bus'].tolist(), carrier=base_data.gen_hps['carrier'].tolist(),
             p_nom=base_data.gen_hps['p_nom'].tolist(), max_hours=base_data.gen_hps['max_hours'].to_list(),
             p_max_pu=base_data.gen_hps['p_max_pu'].tolist(),
             efficiency_dispatch=base_data.gen_hps['efficiency_dispatch'].tolist(),
             standing_loss=base_data.gen_hps['standing_loss'].tolist())

# Batteries
network.madd("StorageUnit", base_data.gen_storage.index, suffix='_Battery',
             bus=base_data.gen_storage['bus'].tolist(), carrier=base_data.gen_storage['carrier'].tolist(),
             p_nom=base_data.gen_storage['p_nom'].tolist(), max_hours=base_data.gen_storage['max_hours'].to_list(),
             p_max_pu=base_data.gen_storage['p_max_pu'].tolist(),
             efficiency_dispatch=base_data.gen_storage['efficiency_dispatch'].tolist(),
             standing_loss=base_data.gen_storage['standing_loss'].tolist())

network.madd("StorageUnit", base_data.gen_batteries_small['bus'], suffix='_Battery_small',
             bus=base_data.gen_batteries_small['bus'].tolist(), carrier=base_data.gen_batteries_small['carrier'].to_list(),
             p_nom=base_data.gen_batteries_small['p_nom'].tolist(), max_hours=base_data.gen_batteries_small['max_hours'].to_list(),
             p_max_pu=base_data.gen_batteries_small['p_max_pu'].tolist(), efficiency_dispatch=base_data.gen_batteries_small['efficiency_dispatch'].tolist(),
             standing_loss=base_data.gen_batteries_small['standing_loss'].tolist(), cyclic_state_of_charge=True)


#  ------------- OPT POWER FLOW ----------------------------------------------
network.lopf(snapshots=network.snapshots, solver_name='cplex', pyomo=False)  


#  ------------- RESULTS -----------------------------------------------------

export_network_results(scenario_name, reference_year, network)
print('All completed :>')
