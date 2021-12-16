# PyPSA-PL

## Introduction

Welcome to the `PyPSA-PL` repository! `PyPSA-PL` is a set of scripts and a dataset that can be used with the `PyPSA` open energy modelling framework to analyze the mechanics of the Polish power system. The model was created by the Instrat Foundation and described in a trilogy of publications about the Polish coal phase-out:
1. Czyżak, P., Wrona, A. (2021). Achieving the goal. Coal phase-out in Polish power sector. Instrat Policy Paper 01/2021. https://instrat.pl/en/coal-phase-out/
2. Czyżak, P., Sikorski, M., Wrona, A. (2021). What’s next after coal? RES potential in Poland. Instrat Policy Paper 06/2021. https://instrat.pl/en/res-potential/
3. Czyżak, P., Wrona, A., Borkowski, M. (2021). The missing element. Energy security considerations. Instrat Policy Paper 09/2021. https://instrat.pl/en/energy-security/

## Citation and license

The recommended citation is:

`Czyżak, P., Mańko, M., Sikorski, M., Stępień, K., Wieczorek, B. (2021). PyPSA-PL - An open energy model of the Polish power sector based on the PyPSA framework. Instrat. https://github.com/instrat-pl/pypsa-pl`

`PyPSA-PL` is based on the [PyPSA](https://pypsa.readthedocs.io/en/latest/index.html) open energy modelling framework used by many research institutions around the world. Whenever using `PyPSA-PL`, please also give credit to the authors of `PyPSA` following the guidelines [described in the PyPSA documentation](https://pypsa.readthedocs.io/en/latest/citing.html).

[Similarly to PyPSA](https://pypsa.readthedocs.io/en/latest/introduction.html?highlight=license#licence), `PyPSA-PL` is distributed under the [MIT license](https://github.com/instrat-pl/pypsa-pl/blob/main/LICENSE).

## Setup

To use the model you will need to have a basic understanding of `Python` and `conda`, an LP solver and a high performance computer with enough memory (32 GB RAM for the simplified 400kV network, 64 GB for the full 220 kV network). It is possible to run the model with an open solver, but the performance might not be satisfactory. In case of lacking hardware resources, it is possible to run the model using a virtual machine e.g. hosted in Azure. You can also run single days instead of a full year to improve solving time.

Before you start working with PyPSA-PL, please follow the [installation guidelines for PyPSA](https://pypsa.readthedocs.io/en/latest/installation.html).

In the PyPSA-PL repository you will find a `requirements.txt` file that can be used to set up a `conda` environment. You will notice that this file downloads the `0.17.0` version of the `pypsa` package - that is the version used to create `PyPSA-PL`. There were significant changes introduced in `PyPSA 0.18.0`, but the PL version has not be upgraded yet. 

The basic installation steps can be summarized as follows:

1. Clone the `PyPSA-PL` repository
2. Create a new `conda` environment (optionaly using the `requirements.txt` file)
3. Install an LP solver (make sure it is available in the `Python` terminal)
4. Install the `PyPSA` package (if not done via the `requirements.txt` file)

## Getting started

`PyPSA-PL` is a set of scripts and spreadsheets that can be used to generate inputs for the actual `PyPSA` framework.
After you complete the installation procedue, it is relatively easy to get some first results:

1. Open the `prepare_input_files` script in the `pypsa-pl` folder using your IDE.
2. Set the `scenario_name` and `reference_year`. The available scenarios are `instrat` and `pep2040`, the input data is provided for years `2020`, `2021`, `2025`, `2030`, `2035`, `2040`. It is necessary to run each year separately. 
3. Save the file and run it (if an error pops up, make sure the terminal points to the correct directory).
4. This will create the `PyPSA` input data in your `pypsa-pl/pypsa/scenario_name/data/reference_year/` folder.
5. Open the `run_pypsa_lopf` script in the `pypsa-pl` folder using your IDE
6. Set the solver name in line 155 (where the `network.lopf` function is called). See the `PyPSA` documentation for a list of compatible solvers.
7. Save the file and run it.
8. Once the optimization is finished, the results will be placed in your `pypsa-pl/pypsa/scenario_name/results/reference_year/` folder.

## Accessing the results

You can look at the basic generation, SRMC and emissions data provided in `csv` files, but the recommended way of processing results is by using the `network.nc` file. To do that, open a jupyter notebook, create a new `pypsa.Network()` object and import the data into it using the `import_from_netcdf()` function. An example is shown below.

Import the network from the `netCDF` file:

```
network_2040 = pypsa.Network()
network_2040.import_from_netcdf('pypsa/instrat/results/2040/network.nc')
```

Access the model outputs, e.g. the generation by carrier:

```
network_2040.generators_t.p.groupby(network_2040.generators.carrier, axis=1).sum().sum()
```

You can find several examples of processing results in the `PyPSA` documentation.

## Model design

Two scenarios are provided with `PyPSA-PL`:

- Instrat - based on Instrat's own unit-by-unit coal phase-out schedule, with detailed year-by-year RES forecasts. The scenario is described in detail in the `What’s next after coal? RES potential in Poland` paper.
- PEP2040 - based on the official `Polish Energy Policy until 2040`. Since that only provides aggregated capacity and generation by fuel values, a unit-by-unit decommissioning schedule for coal plants was generated. The expansion plans for other technologies were proposed as well based on the currently available knowledge (e.g. the placement of gas plants, nuclear units etc.). In this scenario the imports are disabled, following the reasoning described in `PEP2040`.

Please note that `PyPSA-PL` is not configured as a capacity expansion model (CEM), but rather a merit-order style dispatch model. The installed capacity for different technologies is provided manually as part of the scenario design. This decision was made to ensure that the `Instrat` scenario is technically, politically, socially, economically viable in each given year, while also following a dispatching procedure similar to the one performed by the TSO. Running the model as a CEM means the capacity additions are set by the optimizer and do not always correspond to reality. It is possible to set the `capex` values for each technology and convert `PyPSA-PL` to a CEM, but that may require some programming work by the user. 

`PyPSA-PL` contains a full representation of the Polish `400 kV` transmission network and a simplified representation of `220 kV` lines, with a total of 75 nodes. You can also run the model using the full `220 kV` network by using the `lines_380and220` tab in the `lines.xlsx` file and `buses_380and220` in the `buses.xlsx` file, but that will increase the node count to 200 and require at least 64 GB of RAM on your computer.

Each node of the network has some load attached to it, as well as distributed generators (e.g. wind farms, PV) and possibly utility-scale generation units (coal, gas plants etc.). The load and RES spatial resolution is set on a voivodeship level and then disaggraged proportionally onto the nodes in the given voivodeship.

Only the Polish transmission network is modelled, with existing cross-border links provided but no representation of the European transmission network. The link capacities are set to real values, which limits the peak imports. The actual imports are determined by the prices in other countries and these will translate to aggregated net import volumes in a given year. The import prices were calibrated using 2020 values and then adapted to keep them above the SRMC of Polish gas units in the future (it is most likely that if RES generation is low in Poland, it will be low in some neighbouring countries as well and the national gas units will set the merit order price).

## Changing input parameters

To change the scenarios, access the files located in the `inputs` folder. Files shared across scenarios are placed in the main directory, additionally, each scenario will have a dedicated folder.

The scenario-specific files are:

- `cbf_links` - the capacities for cross-border connections. These were designed based on entso-e data about the currently planned network expansion projects.
- `scenario` - the installed capacity for distributed generators, as well as prices, demand projections and constants. This will be the main file to modify while building your own scenario, especially via changing the capacity projections for distributed RES technologies.
- `utility_units` - parameters for large generators like coal units, CHPs, gas plants, offshore wind farms. Modifying this file requires more knowledge - you need to define the parameters for each specific unit.

The shared files are:

- `buses` - a list of network buses, for both the standard 400kV network and the full 220kV network
- `capacity_factors` - the wind and solar generation in each hour of the year (and across voivodeships), the generalized CHP generation profile (to account for the fact that CHP units have limited availability in the summer)
- `buses` - a list of network buses, for both the standard 400kV network and the full 220kV network
- `installed_capacity_distribution` - the spatial placement of distributed generator capacity (onshore wind, PV, small-scale batteries, industrial units) across voivodeships
- `lines` - a list of network lines per year - the lines are expanded according to the TSO's plans for 2030 and Instrat's forecast to accomodate increased RES deployment. See the report: `The missing element` for more details on the methodology.
- `load_decomposition_over_voivodeships_2019` - the distribution of load across Polish voivodeships (this is the most detailed officially provided load disaggregation)
- `load_hour_factors_2019` - the hourly demand profile in 2019 - this is scaled for future years using the yearly demand projections from PEP2040. The year 2019 was chosen because the demand in 2020 was strongly impacted by the COVID-19 pandemic.
- `wojewodztwa-max` - a `.geojson` file with the Polish voivodeships, these are used to map network nodes with demand and RES generation values

Once changes have been made, you need to re-generate the `pypsa` inputs using `prepare_input_files` scipt and then run the optimization using `run_pypsa_lopf`.

You can look at the `pypsa/scenario_name/data` folder contents to diagnose any data issues.

## Further improvements

We are aware that many improvements need to be made in the `PyPSA-PL` model, some of them include:

- Heating - including heat demand in the model and proposing a full coal-to-X conversion scheme for existing CHP plants
- Hydrogen - including electrolyzers in the model, which could allow for a more detailed assessment of using RES energy surplusses for green hydrogen generation
- EU-wide network - improving the representation of the European-wide transmission network and the cross-border flows by adding at least a single node per country, with capacity expansion projections based on e.g. the NECP's
- Coal unit flexibility - adding ramp-up/down and minimum utilization constraints. This was removed from the model due to the significant increase in model complexity and solving times, but it is possible to implement these constraints in base `PyPSA`.
- Demand and RES disaggregation - both load and RES distribution are defined on voivodeship level. These could be disaggregated onto the network nodes using a more sophisticated approach - e.g. taking into account population density, economic activity, industry etc.
- Demand from other sectors - the demand projection for 2020-2040 comes from the Polish TSO (also included in PEP2040). It is very likely that the electrification of transport and heating will lead to even higher electricity demand increases. 
- Line loading - to ensure an even distribution of line loading with the simplified 220 kV network, the resistance and reactance of 220 kV lines was changed in the `line_types` file in core `PyPSA`. This is not an ideal solution and should be handled in the `PyPSA-PL` code.
- Thermal unit maintenance schedule - the Polish coal units are extremely prone to outages due to their age. This is not represented in the model and could mean that the availability of coal plants is higher than in reality. A maintenance schedule and a random outage algorithm could be included to represent the actual working conditions of coal plants.

You are welcome to expand and modify `PyPSA-PL` according to your needs (honoring the license - see `Citiation and license` section). You can also submit pull requests onto the `PyPSA-PL` repository, but due to a lack of resources we cannot ensure we will be able to review and approve those.

## References

Please look at the Instrat Policy Papers cited in the `Introduction` for a detailed list of sources and a description of the methodology, the scenario design assumptions, price parameters etc.

The most significant model references are listed below:

- PyPSA: T. Brown, J. Hörsch, D. Schlachtberger, PyPSA: Python for Power System Analysis, 2018, Journal of Open Research Software, 6(1), arXiv:1707.09913, DOI:10.5334/jors.188
- EMHIRES: Gonzalez Aparicio I; Zucker A; Careri F; Monforti-Ferrario F; Huld T; Badger J. EMHIRES dataset Part I: Wind power generation. European Meteorological derived HIgh resolution RES generation time series for present and future scenarios . EUR 28171 EN. Luxembourg (Luxembourg): Publications Office of the European Union; 2016. JRC103442 and Gonzalez Aparicio, Iratxe; Huld, Thomas; Careri, Francesco; Monforti Ferrario, Fabio; Zucker, Andreas (2017): Solar hourly generation time series at country, NUTS 1, NUTS 2 level and bidding zones. European Commission, Joint Research Centre (JRC) [Dataset] PID: http://data.europa.eu/89h/jrc-emhires-solar-generation-time-series
- Renewables.ninja: Pfenninger, Stefan and Staffell, Iain (2016). Long-term patterns of European PV output using 30 years of validated hourly reanalysis and satellite data. Energy 114, pp. 1251-1265. doi: 10.1016/j.energy.2016.08.060 and Staffell, Iain and Pfenninger, Stefan (2016). Using Bias-Corrected Reanalysis to Simulate Current and Future Wind Power Output. Energy 114, pp. 1224-1239. doi: 10.1016/j.energy.2016.08.068
- Instrat power plant database: Stępień, K., Czyżak, P., Hetmański, M. (2021). Power plant database - Poland. Instrat. http://bit.ly/instratpowerplants
- PEP2040: Dziennik Urzędowy Rzeczypospolitej Polskiej. (2021). Obwieszczenie Ministra Klimatu i Środowiska z dnia 2 marca 2021 r. w sprawie polityki energetycznej państwa do 2040 r. MONITOR
POLSKI: https://monitorpolski.gov.pl/M2021000026401.pdf
- TNDP: PSE. (2020). Plan rozwoju w zakresie zaspokojenia obecnego i przyszłego zapotrzebowania
na energię elektryczną na lata 2021-2030: https://www.pse.pl/documents/20182/21595261/
Dokument_glowny_PRSP_2021-2030_20200528.pdf
- ENTSO-E TYNDP 2020: https://tyndp.entsoe.eu/
- IEA WEO 2021: https://www.iea.org/reports/world-energy-outlook-2021
- GLAES: Ryberg, D. S., Robinius, M., & Stolten, D. (2018). Evaluating Land Eligibility Constraints of Renewable Energy Sources in Europe. Energies 11 (5). pp. 1246. http://www.mdpi.com/1996-1073/11/5/1246 
- energy.instrat.pl
- PSE: https://www.pse.pl/dane-systemowe
- GUS: https://bdl.stat.gov.pl/BDL/start
- ARE: ARE. (2021). Informacja Statystyczna o Energii Elektrycznej. https://www.are.
waw.pl/badania-statystyczne/wynikowe-informacje-statystyczne#informacja-statystyczna-o-energii-elektrycznej
