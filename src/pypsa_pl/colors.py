technology_colors = {
    "": "rgba(255,255,255,0)",
    "Non-RES": "#B5C6CF",
    "Coal and other non-RES": "#B5C6CF",
    "Hard coal": "#B5C6CF",
    "Lignite": "#72A0B7",
    "Natural gas": "#4B7689",
    "Biomass/biogas": "#263b46",
    "Coke-oven gas": "#263b46",
    "Oil": "#263B46",
    "Nuclear": "#263B46",
    "Biomass wood chips": "#AB9D99",
    "Biomass straw": "#816D66",
    "Biogas": "#573C33",
    "Wind": "#69F0AE",
    "Wind onshore": "#B9F6CA",
    "Wind offshore": "#01E677",
    "PV": "#EC8CFD",
    "PV roof": "#F3B3FD",
    "PV ground": "#E040FB",
    "Hydro ROR": "#2979ff",
    "Hydrogen": "#0A1E40",
    "Hydro PSH": "#82B1FF",
    "Hydro PSH/Battery": "#82B1FF",
    "Battery small": "#E3E4E4",
    "Battery large": "#B6B6B7",
    "Battery": "#B6B6B7",
    "DSR": "#1B1C1E",
    "Import": "#767778",
    "Net import": "#767778",
    "Export": "#49494b",
    "Import/export": "#767778",
    "Hydro PSH store": "#82B1FF",
    "Hydro PSH dispatch": "#82B1FF",
    "Battery large store": "#B6B6B7",
    "Battery large dispatch": "#B6B6B7",
    "Battery small store": "#E3E4E4",
    "Battery small dispatch": "#E3E4E4",
    "Total": "#C1843D",
}

technology_names_pl = {
    "": "",
    "Non-RES": "Nie-OZE",
    "Hard coal": "Węgiel kamienny",
    "Coal and other non-RES": "Węgiel i inne nie-OZE",
    "Lignite": "Węgiel brunatny",
    "Natural gas": "Gaz ziemny",
    "Biomass/biogas": "Biomasa/biogaz",
    "Coke-oven gas": "Gaz koksowniczy",
    "Oil": "Ropa naftowa",
    "Nuclear": "Energia jądrowa",
    "Biomass wood chips": "Biomasa drzewna",
    "Biomass straw": "Biomasa agro",
    "Biogas": "Biogaz",
    "Wind": "Wiatr",
    "Wind onshore": "Wiatr - ląd",
    "Wind offshore": "Wiatr - morze",
    "PV": "PV",
    "PV roof": "PV - dach",
    "PV ground": "PV - gruntowe",
    "Hydro PSH": "Woda - ESP",
    "Hydro PSH/Battery": "ESP/baterie",
    "Hydro ROR": "Woda - przepływowe",
    "Hydrogen": "Wodór",
    "Battery small": "Baterie małe",
    "Battery large": "Baterie duże",
    "Battery": "Baterie",
    "DSR": "DSR",
    "Import": "Import",
    "Net import": "Import netto",
    "Export": "Eksport",
    "Import/export": "Import/eksport",
    "Hydro PSH store": "ESP - ładowanie",
    "Hydro PSH dispatch": "ESP - generacja",
    "Battery large store": "Baterie duże - ładowanie",
    "Battery large dispatch": "Baterie duże - generacja",
    "Battery small store": "Baterie małe - ładowanie",
    "Battery small dispatch": "Baterie małe - generacja",
    "Total": "Razem",
}

if __name__ == "__main__":
    import os
    from pypsa_pl.config import data_dir

    os.makedirs(data_dir("clean", "flourish"), exist_ok=True)

    with open(data_dir("clean", "flourish", "colors_en.txt"), "w") as f:
        for tech, color in technology_colors.items():
            if tech == "":
                continue
            f.write(f"{tech}: {color}\n")

    with open(data_dir("clean", "flourish", "colors_pl.txt"), "w") as f:
        for tech, color in technology_colors.items():
            if tech == "":
                continue
            tech_pl = technology_names_pl[tech]
            f.write(f"{tech_pl}: {color}\n")
