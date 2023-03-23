import pandas as pd
from geopy import distance
from shapely.geometry import shape, Point
import json


def assign_buses_to_voivodeships(buses):
    # Get coordinates of voivodeships in json
    voivodeships_geojson = 'inputs/wojewodztwa-max.geojson'
    voivodeships_geojson = json.load(open(voivodeships_geojson, 'r', encoding='utf-8'))

    # Leave only buses names and coordinates
    df_buses = buses
    df_buses = df_buses.set_index('name')
    df_buses = df_buses[['x', 'y']]

    points = [df_buses.iloc[i].to_list() for i in range(len(df_buses))]

    # Check point by point if it is in any voivodeship. If yes, then assign it to this voivodeships
    points_in_vois = {}
    for feature in voivodeships_geojson['features']:
        voi = feature['properties']['nazwa']
        points_in_vois[voi] = {}
        polygon = shape(feature['geometry'])
        for i, [point, name] in enumerate(zip(points, df_buses.index)):
            point_ = Point(point)
            if polygon.contains(point_):
                points_in_vois[voi][name] = point

    return points_in_vois


def create_lines_between_buses(year, df_buses, save_directory):
    buses = df_buses['name'].to_list()

    # Create lines between buses but only for existing ones
    df_lines = pd.read_excel('inputs/lines.xlsx', sheet_name=str(year), index_col=0)
    df_lines['bus0'] = df_lines['bus0'].astype(str)
    df_lines['bus1'] = df_lines['bus1'].astype(str)
    df_lines = df_lines[(df_lines['bus0'].isin(buses)) & (df_lines['bus1'].isin(buses))]

    df_buses.to_csv(f'{save_directory}/buses.csv', index=False)
    df_lines.to_csv(f'{save_directory}/lines.csv', index=False)
